"""Universal standard 3x3 convolution harness (FP32 wide-vector mode).

A single parameterized harness that works for ANY valid standard convolution
configuration (spatial >= 16x16, in_channels >= 1). Same chunk-interleaved
internal layout and FPB=28 super-block kernel packing, no bias/activation
folding (see ``conv_universal_bn_activation`` for the bias+ReLU twin). Runs
on the emulator's wide-vector debug datapath (see
docs/content/wide-vector-debug-mode.md) -- weights and activations are
genuine floats, no INT8 quantization anywhere in this kernel.

Pipeline per filter: r_acc seeded from the first 3x3 conv tap
(``ACC.ADD.FIRST``), then += the remaining taps over all input channels, then
``ACTIVATE identity`` -> store 128 elements. Masking (edge-column zeroing) is
mode-blind and runs identically in wide-vector mode as narrow mode.

Kernel super-block layout (FPB=28):
  One 256-element super-block holds up to 28 input-channel slots of one
  output filter. Channel ``s`` occupies elements ``[s*9 .. s*9 + 9)``:
  28 * 9 = 252 elements <= 256. Channels 0..13 land in the first 128-element
  half (R0), 14..27 in the second (R1); the shared-index ``mult.ve``
  (fixed_idx 0..255) addresses all 28.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp

    app = ConvUniversalApp(
        inst_path="conv_universal.bin",
        input_path="input.bin",      # raw [in_channels, height, width] float32
        kernel=weights_nhwc,         # np.ndarray [out_ch, in_ch, 3, 3] float32
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
    REQUIRES,
    conv_query,
    min_rows_for_chunk_floor,
    next_valid_cols,
    positive_dims,
)
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    pass

# -- Memory layout -----------------------------------------------------------
# XMEM instruction operands (``offset + base`` on LDR_*/store) are ROW
# numbers, not byte addresses: the emulator scales them by the active mode's
# row size (128 B narrow, 512 B wide-vector debug). This app runs FP32
# wide-vector only, so ROW_BYTES is always 512.
#
# r_cyclic operands (MULT.RC.* ``rc_idx`` reads and LDR_CYCLIC_MULT_REG's
# ``index`` writes) are ELEMENT-indexed and the ring is 512 elements
# regardless of mode, so they are already mode-blind and must NOT be
# rescaled by ROW_BYTES.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

OUTPUT_CHUNK_BYTES = CHUNK_ELEMENTS * 4  # bytes per output filter per chunk (FP32)

SUPER_BLOCK_ELEMENTS = 2 * CHUNK_ELEMENTS  # 256 = R0 half + R1 half
SUPER_BLOCK_ROWS = 2                       # the same super-block, in XMEM rows
HALF_FPB = CHUNK_ELEMENTS // 9             # 14: channels per 128-element half (9 taps each)
FPB = 2 * HALF_FPB                         # 28: channels per super-block (R0+R1 shared index)


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

    Mask polarity (matches upstream ``_mult_mask_and_shift``): a mask bit of
    **1 KEEPS** the lane, **0 ZEROES** it.  ``rows_per_chunk`` = 128 // cols
    spatial rows are packed into the 128 lanes; row ``r`` occupies lanes
    ``[r*cols, r*cols + cols)``. This blob is a bitmask, independent of the
    active arithmetic mode -- it does NOT widen with FP32/element size.

    Left/right edge columns are handled at runtime by ``mask_shift`` (with
    ``CR15.partition = cols``), so the slots only zero whole out-of-bounds rows:

      slot 0 (none)       -> KEEP every lane (interior / kr=0-row taps)
      slot 3 (top row)    -> ZERO packed row 0       (g0 section kr=-1 taps)
      slot 6 (bottom row) -> ZERO the last packed row (gN section kr=+1 taps)

    One blob carries all three; the asm selects slot 3 in g0 and slot 6 in gN,
    so no mid-program R_MASK reload is needed.
    """
    rows_per_chunk = 128 // cols
    top_bits = set(range(0, cols))                                  # row 0
    bottom_row = rows_per_chunk - 1
    bottom_bits = set(range(bottom_row * cols, bottom_row * cols + cols))

    # Per slot, the set of lanes to ZERO (bit cleared); all others kept (bit 1).
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


def _pack_conv_weights_fpb28(weights_reordered: np.ndarray) -> bytes:
    """Pack [out_ch, in_ch, 9] float32 (taps already reordered) into FPB=28
    super-blocks.

    Each super-block is 256 ELEMENTS laid out linearly: channel ``s``
    occupies elements ``[s*9 .. s*9+9)``. The first 128 elements are loaded
    into R0 and the second 128 into R1; the asm uses mult.ve with a shared
    fixed_idx (0..255) that sweeps the entire super-block. Per-filter row
    stride: ceil(in_ch / 28) * SUPER_BLOCK_ROWS.
    """
    out_ch, in_ch, k2 = weights_reordered.shape
    if k2 != 9:
        raise ValueError(f"expected last dim=9 (taps), got {k2}")

    super_blocks_per_filter = math.ceil(in_ch / FPB)
    total_elements = out_ch * super_blocks_per_filter * SUPER_BLOCK_ELEMENTS
    packed = np.zeros(total_elements, dtype=np.float32)

    for f in range(out_ch):
        for sb in range(super_blocks_per_filter):
            sb_base = (f * super_blocks_per_filter + sb) * SUPER_BLOCK_ELEMENTS
            for s in range(FPB):
                ic = sb * FPB + s
                if ic >= in_ch:
                    break
                dst = sb_base + s * 9  # linear layout: 0,9,18,...,243
                packed[dst:dst + 9] = weights_reordered[f, ic, :]
    return packed.tobytes()


class ConvUniversalApp(IpuApp):
    """Universal standard 3x3 convolution application harness (FP32).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary, raw ``[in_channels, height,
                      width]`` float32.
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous float32 file. Reshaped to ``[out_ch,
                      in_ch, 3, 3]`` and packed the same way.
        output_path:  Optional path to write output (raw ``[out_channels,
                      height, width]`` float32).
        height:       Spatial height (>= 1). Padded internally to satisfy
                      the hardware's cols-in-{16,32,64,128} constraint and
                      the num_chunks>=2 floor.
        width:        Spatial width (>= 1). Padded internally the same way.
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
        # cols must be in {16,32,64,128} (one packed row per mask-partition
        # group); rows*cols must be a whole number >= 2 of 128-element
        # chunks. Padded internally -- see _spec_support.next_valid_cols /
        # min_rows_for_chunk_floor.
        cols = next_valid_cols(width)
        rows = min_rows_for_chunk_floor(height, cols)
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_chunks = (rows * cols) // CHUNK_ELEMENTS
        # Row-granular strides (see "Row addressing" above). One chunk is one
        # XMEM row, so a group of ``in_channels`` chunks is ``in_channels`` rows.
        self.in_group_stride = in_channels
        self.blocks_per_filter = math.ceil(in_channels / FPB)
        # A super-block spans SUPER_BLOCK_ROWS (= 2) rows: the R0 half + R1 half.
        self.total_kernel_rows = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_ROWS
        )
        self.total_kernel_elements = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_ELEMENTS
        )

        # -- Dynamic region layout -------------------------------------------
        # Size each region from THIS configuration instead of fixed gaps (see
        # convolutions_universal/__init__.py's allocate_regions docstring).
        #
        # Input sits one group stride above its own region base so the g0
        # kr=-1 prefetch (offset lr_chunk_base - cr6) bottoms out at exactly
        # that base rather than underflowing -- so the "input" region's real
        # size is the headroom PLUS the data, not just the data. Region sizes
        # are in ELEMENTS; setup() scales to bytes via ROW_BYTES (FP32,
        # 4 B/element, always).
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
        # Tap order in the walking-pointer asm: kr=-1 → kr=0 → kr=+1, with
        # kc=-1 → 0 → +1 within each row.  That's natural row-major from the
        # source [out_ch, in_ch, 3, 3] — no reordering needed.
        w_reordered = weights.reshape(self.out_channels, self.in_channels, 9)
        return _pack_conv_weights_fpb28(w_reordered)

    @staticmethod
    def make_state() -> IpuState:
        """Build the FP32 wide-vector state this app requires (see
        pointwise_conv_unified.make_state for the same convention)."""
        return IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
            wide_vector_quantize_output=False,
        )

    def setup(self, state: "IpuState") -> None:
        # input_path holds the TRUE (unpadded) [in_channels, height, width]
        # float32 tensor -- pad + chunk-pack to the on-device layout here.
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.in_channels, self.height, self.width)
        padded = np.zeros((self.in_channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_data_row * ROW_BYTES, input_data)

        # Pack and load kernel (dense FPB=28 super-block layout)
        kernel_packed = self._pack_kernel()
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        # Border masks: a SINGLE blob carrying all 3 slots (0=none, 3=top-row
        # zero, 6=bottom-row zero), loaded once at init.  The g0 section selects
        # slot 3, the gN section selects slot 6 — no mid-program R_MASK reload.
        # Vertical out-of-bounds rows are masked (no zero chunk in the cyclic
        # register); left/right edge columns are applied at runtime by mask_shift
        # (see CR15 partition below). The mask blob does NOT widen -- it is 1
        # bit per lane regardless of element width -- only its row address
        # scales.
        state.xmem.write_address(
            self.mask_base_row * ROW_BYTES, build_border_mask_blob(self.cols)
        )

        # CR15 dstructure: partition so each partition group is exactly one
        # packed spatial row (group size == cols).  The asm's mask_shift then
        # injects the left/right edge-column zero at each packed-row boundary.
        cols_to_partition = {
            128: Partition.P0,  # 1 group of 128 lanes (one packed row per chunk)
            64: Partition.P2,   # 2 groups of 64 lanes
            32: Partition.P4,   # 4 groups of 32 lanes
            16: Partition.P8,   # 8 groups of 16 lanes
        }
        state.set_cr_dstructure(
            valid_elements=128,
            partition=cols_to_partition[self.cols],
        )

        # CR register map — adapted to master ISA:
        #   CR0 is read-only constant 0, CR1 is read-only constant 1, CR15 is the
        #   dstructure (valid_elements | partition). So the input/kernel bases that
        #   used to live in CR0/CR1 were relocated:
        #     CR10 = INPUT_BASE_ROW   (was CR0; CR0 now serves the zero-constant role)
        #     CR5  = KERNEL_BASE_ROW  (was CR1; CR5's old num_chunks value is unused in asm)
        # All four are XMEM *row* numbers, not byte addresses -- the emulator
        # scales them per active mode.
        state.regfile.set_cr(10, self.input_base_row)
        state.regfile.set_cr(5, self.kernel_base_row)
        state.regfile.set_cr(2, self.output_base_row)
        state.regfile.set_cr(3, self.mask_base_row)      # single mask blob (slots 0/3/6)

        # Set parameter CR registers
        state.regfile.set_cr(4, self.cols)
        # cr6/cr7/cr8 are XMEM-space and therefore row counts. cr7 bounds
        # lr_ch_ctr, which is added to lr_chunk_base to form the input-row
        # offset, so it must share lr_ch_ctr's unit (rows, not bytes).
        state.regfile.set_cr(6, self.in_group_stride)
        state.regfile.set_cr(7, FPB)                # channel group = 28 rows
        state.regfile.set_cr(8, self.total_kernel_rows)
        # cr11 = chunk-loop limit = (num_chunks - 1) * in_group_stride, in ROWS
        # (in_group_stride is row-granular). Used by asm to compare lr8 (chunk
        # base row) against the chunk limit, replacing the old lr9 counter.
        # Biased by one in_group_stride to match lr_chunk_base's guard offset.
        state.regfile.set_cr(
            11, (self.num_chunks - 1) * self.in_group_stride + self.in_group_stride
        )

        # cr12/cr9 keep their R_CYCLIC-ELEMENT meaning (slot stride 128, ring
        # advance 384). The ring is 512 ELEMENTS regardless of mode, so these
        # are already mode-blind and must NOT be divided by CHUNK_ELEMENTS.
        #
        # Their former *XMEM* role (advance one chunk = one row) is now served
        # by the read-only constant CR1 (= 1 row), since in row space a chunk
        # stride is literally 1. That split is why cr12 is no longer overloaded.
        state.regfile.set_cr(12, CHUNK_ELEMENTS)     # 128 ELEMENTS (r_cyclic slot stride)
        state.regfile.set_cr(13, SUPER_BLOCK_ROWS)   # 2 ROWS (kernel super-block stride)
        # cr9 = ring advance = 3 * 128 = 384 ELEMENTS.  9-cyc role-rotating scheme:
        # lr_read (kr=0 slot) advances -128 (= +384 mod 512) per channel.
        state.regfile.set_cr(9, 3 * CHUNK_ELEMENTS)  # 384 ELEMENTS
        # cr14 = end-of-9 walking-pointer wrap step: brings lr_walk from this ch's
        # tap-9 offset (lr_read + cols + 1) to next ch's tap-1 offset
        # ((lr_read - CHUNK_ELEMENTS) - cols - 1).  Under -128 rotation this is
        # +(RING_ADV - 2*cols - 2) with RING_ADV = 384 (= -128 mod 512 + 512).
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


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: kernel_size==3,
# groups==1 (plain conv; depthwise has its own kernel), stride==1, padding==1
# ("same" padding for a 3x3 kernel, the only mode this app's masking scheme
# implements). See _spec_support.ConvQuery for the full query shape.


def _supports(**params):
    q = conv_query(**params)
    if bad := positive_dims(q):
        return no(bad)
    if q.kernel_size != 3:
        return no(f"handles only kernel_size=3; got {q.kernel_size}")
    if q.dilation != 1:
        return no(f"handles only dilation=1; got {q.dilation}")
    if q.padding != 1:
        return no(f"handles only padding=1 (\"same\" padding for a 3x3 kernel); got {q.padding}")
    if q.stride != 1:
        return no(f"handles only stride=1; got {q.stride}")
    if q.groups != 1:
        return no(f"handles only groups=1 (plain conv); got {q.groups} (depthwise has its own kernel)")
    if q.apply_relu:
        return no("apply_relu=True has no matching app here; see conv_universal_bn_activation")
    if q.has_bias:
        return no("bias is not supported by this kernel; see conv_universal_bn_activation")
    if q.width > 128:
        return no(
            f"width ({q.width}) exceeds 128, the largest width this app "
            "supports; see conv_universal_wide384 for wider images"
        )
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {
        "height": q.height, "width": q.width,
        "in_channels": q.in_channels, "out_channels": q.out_channels,
    }


def _explain(**params):
    q = conv_query(**params)
    cols = next_valid_cols(q.width)
    rows = min_rows_for_chunk_floor(q.height, cols)
    return (
        f"kernel_size=3, groups=1, stride=1, padding=1: the universal 3x3 "
        f"conv kernel (FP32). {q.height}x{q.width} pads internally to "
        f"{rows}x{cols}."
    )


def _caveats(**params):
    q = conv_query(**params)
    cols = next_valid_cols(q.width)
    rows = min_rows_for_chunk_floor(q.height, cols)
    caveats = (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
    )
    if (rows, cols) == (q.height, q.width):
        return caveats
    real = q.height * q.width
    padded = rows * cols
    return caveats + (
        f"{q.height}x{q.width} pads to {rows}x{cols}, so "
        f"{padded - real} of every {padded} spatial positions idle "
        f"({real / padded:.0%} utilisation).",
    )


SPEC = KernelSpec(
    name="conv_universal",
    op="conv2d",
    variant="standard",
    app_class=ConvUniversalApp,
    asm="conv_universal.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
