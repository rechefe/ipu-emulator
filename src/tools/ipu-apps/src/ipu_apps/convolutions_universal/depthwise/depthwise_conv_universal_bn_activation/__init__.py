"""Universal depthwise 3x3 convolution + folded-bias + ReLU harness (FP32).

Derived from ``depthwise_conv_universal``. Same chunk-interleaved I/O layout and
walking-pointer / rotating-cyclic-slot pipeline, with three additions
(mirroring ``conv_universal_bn_activation``):

  * **Folded bias** — one float32 bias per channel, injected as a single extra
    "multiply by 1" accumulate (``acc.first``) at the start of each channel.
    Batch-norm is assumed already folded into the depthwise weights + this bias.
  * **ReLU activation** — applied via ``ACTIVATE relu``.
  * **Mask-based borders** — the top/bottom out-of-bounds rows are zeroed with
    a single 3-slot mask blob (slots 0/3/6) instead of loading a zero chunk into
    the cyclic register (no zero region); left/right edge columns are applied at
    runtime by mask_shift (CR15 partition = cols), mirroring conv.

Per-channel budget: **11 cyc/ch** = 1 bias-seed cycle + 9 weight taps + 1
standalone ACTIVATE cycle (the base app runs 9 cyc/ch with no bias). Runs on
the emulator's wide-vector debug datapath (FP32) -- weights, bias, and
activations are genuine floats, no INT8 quantization anywhere in this kernel.

Kernel super-block layout (FPB=25, stride 10):
  Depthwise produces one output PER channel, so each channel needs its OWN bias
  element — conv's "one bias element shared across a super-block's in-channels"
  does not apply. Instead each channel occupies a **10-element slot**: element 0
  = its float32 bias, elements 1..9 = its 9 weight taps. 25 channels * 10 = 250
  <= 256, so one 256-element super-block (R0 = elements 0..127, R1 = 128..255)
  holds 25 channels. The shared ``mult.ve`` fixed_idx (0..255) addresses both
  halves transparently.

  The asm walks one continuous kernel element index ``lr6`` at +1 per cycle:
  for channel ``s`` the bias-seed reads ``fixed_idx = s*10`` (bias), then the 9
  taps read ``s*10 + 1 .. s*10 + 9``; the next channel's bias is the following
  element, so the 10-cycle/channel body advances ``lr6`` by exactly one channel
  stride.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation import (
        DepthwiseConvUniversalBnActivationApp,
    )

    app = DepthwiseConvUniversalBnActivationApp(
        inst_path="depthwise_conv_universal_bn_activation.bin",
        input_path="input.bin",       # raw [channels, height, width] float32
        kernel=weights,                # np.ndarray [channels, 3, 3] float32
        bias=bias,                     # np.ndarray [channels], float32 (defaults zeros)
        output_path="output.bin",
        height=64, width=64, channels=256,
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
# Reuse the conv_universal_bn_activation mask-blob builder so the two apps share
# one border-mask implementation: a single 128-byte blob (slots 0/3/6) where
# left/right edge columns are applied at runtime via mask_shift (CR15 partition).
from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    build_border_mask_blob,
)
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    pass

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): XMEM offset/base operands on LDR_*/STR_* are
# ROW numbers, not byte addresses. This app runs FP32 wide-vector only, so
# ROW_BYTES is always 512. r_cyclic index/rc_idx operands (lr5, lr3/lr4) stay
# ELEMENT-addressed and are untouched by this migration.

ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

OUTPUT_CHUNK_BYTES = CHUNK_ELEMENTS * 4  # bytes per output channel per chunk (FP32)

FPB = 25            # channels per 256-element super-block (1 bias + 9 taps each)
CH_SLOT_ELEMENTS = 10  # per-channel slot: element 0 = bias, 1..9 = 9 taps
SUPER_BLOCK_ELEMENTS = 256
SUPER_BLOCK_ROWS = SUPER_BLOCK_ELEMENTS * 4 // ROW_BYTES  # = 2


def _pack_depthwise_kernel_bias(
    weights: np.ndarray, bias: np.ndarray, channels: int,
) -> bytes:
    """Pack per-channel (bias + 9 weight taps) into FPB=25 super-blocks (float32).

    Input:  ``weights`` = [channels, 9] float32; ``bias`` = [channels] float32.
    Output: ceil(channels/25) super-blocks of ``256`` elements each.

    Within one super-block, channel ``s`` (0..24) occupies ELEMENTS
    ``[s*10 .. s*10 + 10)``: element ``s*10`` = bias, ``s*10+1 .. s*10+9`` =
    taps. 25*10 = 250 <= 256.  R0 holds elements 0..127, R1 holds 128..255;
    the shared-index ``mult.ve`` (fixed_idx 0..255) spans both halves.
    """
    num_blocks = math.ceil(channels / FPB)
    total_elements = num_blocks * SUPER_BLOCK_ELEMENTS
    packed = np.zeros(total_elements, dtype=np.float32)

    for sb in range(num_blocks):
        sb_base = sb * SUPER_BLOCK_ELEMENTS
        for s in range(FPB):
            ch = sb * FPB + s
            if ch >= channels:
                break
            slot = sb_base + s * CH_SLOT_ELEMENTS
            packed[slot] = bias[ch]
            packed[slot + 1:slot + 10] = weights[ch]
    return packed.tobytes()


class DepthwiseConvUniversalBnActivationApp(IpuApp):
    """Universal depthwise 3x3 convolution + folded-bias + ReLU harness (FP32).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    ``input_path``/``output_path`` hold the TRUE (unpadded) tensor -- see
    ``pointwise_conv_unified``'s class docstring for the exact file-layout
    contract this mirrors.

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary, raw ``[channels, height,
                      width]`` float32.
        kernel:       Numpy weights of shape ``[channels, 3, 3]`` float32.
        kernel_path:  Alternative: path to a raw ``[channels, 9]`` contiguous
                      float32 file.
        bias:         Per-channel float32 bias, shape ``[channels]``. Added
                      once to the accumulator before ReLU. Defaults to zeros.
        output_path:  Optional path to write output (raw ``[channels, height,
                      width]`` float32).
        height:       Spatial height (>= 1). Padded internally.
        width:        Spatial width (>= 1). Padded internally.
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        channels: int,
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

        if height < 1:
            raise ValueError(f"height must be >= 1, got {height}")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        if bias is None:
            bias = np.zeros(channels, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (channels,):
            raise ValueError(f"bias must have shape ({channels},), got {bias.shape}")
        self._bias_array = bias

        self.height = height
        self.width = width
        cols = next_valid_cols(width)
        rows = min_rows_for_chunk_floor(height, cols)
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = (rows * cols) // CHUNK_ELEMENTS
        self.group_stride = channels
        self.num_super_blocks = math.ceil(channels / FPB)
        self.total_kernel_rows = self.num_super_blocks * SUPER_BLOCK_ROWS
        self.total_kernel_elements = self.num_super_blocks * SUPER_BLOCK_ELEMENTS

        input_region_rows = self.group_stride + self.num_chunks * self.group_stride
        self._regions = allocate_regions([
            ("input", input_region_rows * CHUNK_ELEMENTS),
            ("kernel", self.total_kernel_elements),
            ("mask", CHUNK_ELEMENTS),
            ("output", self.num_chunks * self.channels * CHUNK_ELEMENTS),
        ])
        self.input_base_row = self._regions["input"] // CHUNK_ELEMENTS
        self.kernel_base_row = self._regions["kernel"] // CHUNK_ELEMENTS
        self.mask_base_row = self._regions["mask"] // CHUNK_ELEMENTS
        self.output_base_row = self._regions["output"] // CHUNK_ELEMENTS
        self.output_base_addr = self._regions["output"] * 4  # bytes, FP32
        self.input_data_row = self.input_base_row + self.group_stride

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = np.asarray(self._kernel_array, dtype=np.float32)
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.channels * 9 * 4
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, "
                    f"expected {expected} (channels * 9 * 4 B float32)"
                )
            weights = (
                np.frombuffer(raw, dtype=np.float32).reshape(self.channels, 3, 3)
            )
        w_flat = weights.reshape(self.channels, 9)
        return _pack_depthwise_kernel_bias(w_flat, self._bias_array, self.channels)

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
        input_raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        input_chw = input_raw.reshape(self.channels, self.height, self.width)
        padded = np.zeros((self.channels, self.rows, self.cols), dtype=np.float32)
        padded[:, :self.height, :self.width] = input_chw
        input_data = pack_input_chunked(padded, self.cols)
        state.xmem.write_address(self.input_data_row * ROW_BYTES, input_data)

        kernel_packed = self._pack_kernel()
        state.xmem.write_address(self.kernel_base_row * ROW_BYTES, kernel_packed)

        state.xmem.write_address(self.mask_base_row * ROW_BYTES, build_border_mask_blob(self.cols))

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
        state.regfile.set_cr(2, (self.output_base_row - 1) & 0xFFFFFFFF)
        state.regfile.set_cr(3, self.mask_base_row)
        state.regfile.set_cr(9, 384)

        state.regfile.set_cr(4, self.cols)
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, FPB)
        state.regfile.set_cr(8, self.total_kernel_rows)
        state.regfile.set_cr(
            11, (self.num_chunks - 1) * self.group_stride + self.group_stride,
        )
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            raw = state.xmem.read_address(self.output_base_addr, total_outputs * ROW_BYTES)
            padded_out = unpack_output_chunked(raw, self.channels, self.rows, self.cols)
            out = padded_out[:, :self.height, :self.width]
            self.output_path.write_bytes(out.astype(np.float32).tobytes())

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------


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
    if not q.is_depthwise:
        return no(f"handles only depthwise (groups==in_channels); got groups={q.groups}, in_channels={q.in_channels}")
    if q.out_channels != q.in_channels:
        return no(f"depthwise requires out_channels==in_channels; got {q.out_channels} vs {q.in_channels}")
    if not q.apply_relu:
        return no("this kernel always applies ReLU; see depthwise_conv_universal for the plain twin")
    if not q.has_bias:
        return no("this kernel requires bias (folded); see depthwise_conv_universal for the bias-free twin")
    return yes()


def _build(**params):
    q = conv_query(**params)
    return {"height": q.height, "width": q.width, "channels": q.in_channels}


def _explain(**params):
    q = conv_query(**params)
    cols = next_valid_cols(q.width)
    rows = min_rows_for_chunk_floor(q.height, cols)
    return (
        f"kernel_size=3, groups=in_channels (depthwise), stride=1, padding=1, "
        f"bias+ReLU: the universal depthwise 3x3 conv kernel (FP32). "
        f"{q.height}x{q.width} pads internally to {rows}x{cols}."
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
    name="depthwise_conv_universal_bn_activation",
    op="conv2d",
    variant="depthwise_bn_activation",
    app_class=DepthwiseConvUniversalBnActivationApp,
    asm="depthwise_conv_universal_bn_activation.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
