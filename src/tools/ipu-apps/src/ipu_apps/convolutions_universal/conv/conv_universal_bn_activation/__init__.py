"""Universal standard 3x3 convolution + folded-bias + ReLU harness (FP32).

Derived from ``conv_universal``. Same chunk-interleaved internal layout and
FPB=28 super-block kernel packing, on the FP32 wide-vector debug datapath
(see docs/content/wide-vector-debug-mode.md), with two additions:

  * **Folded bias** — one float32 bias per output filter, injected as a
    single extra "multiply by 1" accumulate at the start of each filter (see
    below). Batch-norm is assumed already folded into the conv weights +
    this bias, so no separate BN step is needed.
  * **ReLU activation** — applied via ``ACTIVATE relu`` (instead of identity).

Pipeline per filter: ``r_acc = bias`` (seed), then += 3x3 conv over all
input channels, then ``ACTIVATE relu`` -> store 128 elements.

Kernel super-block layout (FPB=28, +1 bias element):
  One 256-element super-block holds up to 28 input-channel slots of one
  output filter. **Element 0 of every super-block is reserved for the
  filter's bias**; channel ``s`` occupies elements ``[1 + s*9 .. 1 + s*9 +
  9)``. 1 + 28*9 = 253 elements <= 256, so capacity is unchanged. The bias
  element is replicated into every super-block of a filter for a uniform +1
  weight offset, but the asm reads it (and accumulates the bias) only once
  per filter, from super-block 0. Channels 0..13 land in the first
  128-element half (R0), 14..27 in the second (R1); the shared-index
  ``mult.ve`` (fixed_idx 0..255) addresses all 28.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
        ConvUniversalBnActivationApp,
    )

    app = ConvUniversalBnActivationApp(
        inst_path="conv_universal_bn_activation.bin",
        input_path="input.bin",      # raw [in_channels, height, width] float32
        kernel=weights_nhwc,         # np.ndarray [out_ch, in_ch, 3, 3] float32
        bias=bias_f32,               # np.ndarray [out_ch], folded BN bias
        output_path="output.bin",
        height=32, width=32, in_channels=16, out_channels=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    CHUNK_ELEMENTS,
    dump_outputs,
    allocate_regions,
    pack_input_chunked,
    unpack_output_chunked,
)
from ipu_apps.convolutions_universal._spec_support import (
    min_rows_for_chunk_floor,
    next_valid_cols,
)

if TYPE_CHECKING:
    pass

# -- Memory layout -----------------------------------------------------------
# See conv_universal's identical comment for the underflow-avoidance
# rationale (input sits one group stride above its own region base).
# ROW_BYTES is always 512 (FP32 wide-vector, 4 B/element) -- this app has no
# narrow mode.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

OUTPUT_CHUNK_BYTES = CHUNK_ELEMENTS * 4  # bytes per output filter per chunk (FP32)

SUPER_BLOCK_ELEMENTS = 2 * CHUNK_ELEMENTS  # 256 = R0 half + R1 half
SUPER_BLOCK_ROWS = 2                       # the same super-block, in XMEM rows
HALF_FPB = CHUNK_ELEMENTS // 9             # 14: channels per 128-element half (9 taps each)
FPB = 2 * HALF_FPB                         # 28: channels per super-block (R0+R1 shared index)

BIAS_ELEMENT_OFFSET = 1  # super-block element 0 is the bias; channels start at element 1

# Mask slot assignment — a single R_MASK blob (loaded once at init) carries all
# three slots the asm needs.  Left/right edge columns are applied by mask_shift,
# NOT by slots; the slots only zero whole out-of-bounds rows:
#   0 = none        (KEEP all)             -> interior / kr=0-row taps
#   3 = top-row     (zero packed row 0)    -> g0 section kr=-1 taps
#   6 = bottom-row  (zero last packed row) -> gN section kr=+1 taps
# The g0 section selects slot 3, the gN section selects slot 6 — no reload.
MASK_SLOT_NONE = 0
MASK_SLOT_TOP = 3
MASK_SLOT_BOTTOM = 6

def build_border_mask_blob(cols: int) -> bytes:
    """Build the single 128-byte (8 x 16-byte slot) R_MASK blob.

    See conv_universal's identical function for the full explanation. This
    is a bitmask, independent of the active arithmetic mode -- it does NOT
    widen with FP32/element size.
    """
    rows_per_chunk = 128 // cols
    top_bits = set(range(0, cols))                                  # row 0
    bottom_row = rows_per_chunk - 1
    bottom_bits = set(range(bottom_row * cols, bottom_row * cols + cols))

    zero_lanes = {
        MASK_SLOT_NONE: set(),
        MASK_SLOT_TOP: top_bits,
        MASK_SLOT_BOTTOM: bottom_bits,
    }

    mask = bytearray(128)
    for slot, zeros in zero_lanes.items():
        for bit in range(128):
            if bit not in zeros:
                byte_idx = slot * 16 + bit // 8
                mask[byte_idx] |= 1 << (bit % 8)
    return bytes(mask)

def _pack_conv_weights_fpb28(
    weights_reordered: np.ndarray, bias: np.ndarray,
) -> bytes:
    """Pack [out_ch, in_ch, 9] float32 (taps reordered) + per-filter bias
    into FPB=28 blocks.

    Element 0 of every 256-element super-block holds the filter's float32
    bias; channel ``s`` occupies elements ``[1 + s*9 .. 1 + s*9 + 9)``
    (uniform +1 offset). The bias element is replicated into every
    super-block of a filter so the asm's weight index is the same across
    blocks; the asm only accumulates the bias once per filter (from block
    0). 1 + 28*9 = 253 <= 256, capacity unchanged.

    Per-filter row stride: ceil(in_ch / 28) * SUPER_BLOCK_ROWS.
    """
    out_ch, in_ch, k2 = weights_reordered.shape
    if k2 != 9:
        raise ValueError(f"expected last dim=9 (taps), got {k2}")
    if bias.shape != (out_ch,):
        raise ValueError(f"bias must have shape ({out_ch},), got {bias.shape}")

    super_blocks_per_filter = math.ceil(in_ch / FPB)
    total_elements = out_ch * super_blocks_per_filter * SUPER_BLOCK_ELEMENTS
    packed = np.zeros(total_elements, dtype=np.float32)

    for f in range(out_ch):
        for sb in range(super_blocks_per_filter):
            sb_base = (f * super_blocks_per_filter + sb) * SUPER_BLOCK_ELEMENTS
            packed[sb_base] = bias[f]  # element 0 = filter bias
            for s in range(FPB):
                ic = sb * FPB + s
                if ic >= in_ch:
                    break
                dst = sb_base + BIAS_ELEMENT_OFFSET + s * 9  # 1,10,19,...
                packed[dst:dst + 9] = weights_reordered[f, ic, :]
    return packed.tobytes()

class ConvUniversalBnActivationApp(IpuApp):
    """Universal 3x3 convolution + folded-bias + ReLU application harness (FP32).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary, raw ``[in_channels, height,
                      width]`` float32.
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous float32 file.
        bias:         Per-output-channel float32 bias, shape ``[out_ch]``.
                      Added once to the accumulator before ReLU. Defaults to
                      zeros.
        output_path:  Optional path to write output.
        height:       Spatial height (>= 1). Padded internally.
        width:        Spatial width (>= 1). Padded internally.
        in_channels:  Number of input channels (>= 1).
        out_channels: Number of output channels (>= 1).
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        kernel: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if bias is None:
            bias = np.zeros(out_channels, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (out_channels,):
            raise ValueError(
                f"bias must have shape ({out_channels},), got {bias.shape}"
            )
        self._bias_array = bias

        if in_channels < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= 1")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")
        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self.height = height
        self.width = width
        cols = next_valid_cols(width)
        rows = min_rows_for_chunk_floor(height, cols)
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_chunks = (rows * cols) // CHUNK_ELEMENTS
        self.in_group_stride = in_channels
        self.blocks_per_filter = math.ceil(in_channels / FPB)
        self.total_kernel_rows = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_ROWS
        )
        self.total_kernel_elements = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_ELEMENTS
        )

        # -- Dynamic region layout -------------------------------------------
        # Bias is folded into the kernel super-block here (see _pack_kernel),
        # so there is no separate bias region -- 4 regions, same as the base
        # app. See conv_universal's identical comment for the underflow
        # rationale of the input-region headroom.
        input_region_rows = self.in_group_stride + self.num_chunks * self.in_group_stride
        self._regions = allocate_regions([
            ("input", input_region_rows * CHUNK_ELEMENTS),
            ("kernel", self.total_kernel_elements),
            ("mask", CHUNK_ELEMENTS),
            ("output", self.num_chunks * out_channels * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
        self.output_base_row = self._regions["output"] // CHUNK_ELEMENTS
        self.output_base_addr = self._regions["output"] * 4  # bytes, FP32
        self.input_data_row = self.input_base_row + self.in_group_stride

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = np.asarray(self._kernel_array, dtype=np.float32)
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.out_channels * self.in_channels * 9 * 4
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, "
                    f"expected {expected} (out_ch * in_ch * 9 * 4 B float32)"
                )
            weights = (
                np.frombuffer(raw, dtype=np.float32)
                .reshape(self.out_channels, self.in_channels, 3, 3)
            )
        w_reordered = weights.reshape(self.out_channels, self.in_channels, 9)
        return _pack_conv_weights_fpb28(w_reordered, self._bias_array)

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires."""
        return IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )

    def setup(self, state: "IpuState") -> None:
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.in_channels, self.height, self.width)
        padded = np.zeros((self.in_channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_data_row * ROW_BYTES, input_data)

        kernel_packed = self._pack_kernel()
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        state.xmem.write_address(
            self.mask_base_row * ROW_BYTES, build_border_mask_blob(self.cols)
        )

        cols_to_partition = {
            128: Partition.P0,
            64: Partition.P2,
            32: Partition.P4,
            16: Partition.P8,
        }
        state.set_cr_dstructure(
            valid_elements=128,
            partition=cols_to_partition[self.cols],
        )

        state.regfile.set_cr(10, self.input_base_row)
        state.regfile.set_cr(5, self.kernel_base_row)
        state.regfile.set_cr(2, self.output_base_row)
        state.regfile.set_cr(3, self.mask_base_row)

        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, FPB)
        state.regfile.set_cr(8, self.total_kernel_rows)
        state.regfile.set_cr(
            11, (self.num_chunks - 1) * self.in_group_stride + self.in_group_stride
        )

        state.regfile.set_cr(12, CHUNK_ELEMENTS)
        state.regfile.set_cr(13, SUPER_BLOCK_ROWS)
        state.regfile.set_cr(9, 3 * CHUNK_ELEMENTS)
        state.regfile.set_cr(14, (3 * CHUNK_ELEMENTS - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.out_channels
            raw = state.xmem.read_address(
                self.output_base_addr, total_outputs * ROW_BYTES
            )
            padded_out = unpack_output_chunked(raw, self.out_channels, self.rows, self.cols)
            out = padded_out[:, :self.height, :self.width]
            self.output_path.write_bytes(out.astype(np.float32).tobytes())

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)

