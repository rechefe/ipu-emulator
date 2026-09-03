"""First-layer 3x3 convolution + folded-bias + ReLU: 256x256x3 -> 128x128x16 (FP32).

Hardcoded shapes for the network's first layer (stride 2, pad 1):

    Input:  256x256x3   (float32, raw [in_channels, height, width]).
    Kernel: 16 filters x 3 channels x 3x3 taps, packed by the harness.
    Output: 128x128x16  (float32, raw [out_channels, height, width]).

``input_path``/``output_path`` hold the TRUE, unpacked ``[channels, height,
width]`` tensor -- see ``pointwise_conv_unified``'s class docstring for the
exact file-layout contract this mirrors. Internally the harness repacks to
the row-interleaved-by-channel on-device layout the asm expects: (row r,
channel ch) at element (r*3 + ch)*256, and unpacks the output back from (row
r, filter f) at (r*16 + f)*128.

This is the ``_bn_activation`` flavour: per-output-filter **BN bias** is folded
into the kernel (element 0 of each filter's 128-element block) and accumulated
once before the conv taps; **ReLU** is applied at the end. Batch-norm scale is
assumed folded into the conv weights, so only the bias remains separate. Runs
on the emulator's wide-vector debug datapath (FP32) -- weights and
activations are genuine floats, no INT8 quantization anywhere in this kernel.

The 256-wide input strip trick (no seam):
  Each channel-row is 256 wide = two 128-lane R_CYCLIC slots (slot 0 = cols
  0..127, slot 1 = cols 128..255), loaded back-to-back so the full row is one
  contiguous 256-element strip in R_CYCLIC. A kc=-1/0/+1 tap is then just a
  cyclic read at ``center-1 / center / center+1`` — there is no half seam:
  output col 64's center is position 128 and its left neighbour is position 127
  (last element of slot 0), read with no wrap. ``ACC.STRIDE`` decimates the
  256-wide strip to 128 stride-2 outputs.

Border handling (true pad=1):
  * top row (out row 0): kr=-1 input row is out of bounds -> skipped;
  * bottom row (out row 127): center 254, kr=+1 -> row 255 (in bounds) -> no
    special case for 256 -> 128 stride 2;
  * left edge (col 0): kc=-1 has no left neighbour -> position 0 masked;
  * right edge (col 255): kc=+1 has no right neighbour -> position 255 masked.
  Positions 0 and 255 are lane 0 of slot 0 and lane 127 of slot 1; the cyclic
  wrap otherwise pulls the opposite edge into them, so both are mask-zeroed.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_first_layer import (
        ConvFirstLayerApp,
    )

    app = ConvFirstLayerApp(
        inst_path="conv_first_layer.bin",
        input_path="input.bin",   # raw [3, 256, 256] float32
        kernel=weights,           # np.ndarray [16, 3, 3, 3]  (out,in,kh,kw)
        bias=bias,                # np.ndarray [16], folded BN bias, float32
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal._spec_support import REQUIRES, conv_query, positive_dims
from ipu_apps.kernel_registry import KernelSpec, no, yes

if TYPE_CHECKING:
    pass

# -- Fixed shapes ------------------------------------------------------------

IN_ROWS = 256
IN_COLS = 256
IN_CHANNELS = 3
OUT_ROWS = 128
OUT_COLS = 128
OUT_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 2

# Row-interleaved input: (row r, channel ch) at element (r*IN_CHANNELS + ch)*IN_COLS.
IN_ROW_GROUP = IN_CHANNELS * IN_COLS   # 768 elements: one spatial row, all 3 channels
CHANNEL_STRIDE = IN_COLS               # 256: ch-to-ch step within a row group

TAPS_PER_CHANNEL = KERNEL_SIZE * KERNEL_SIZE          # 9
TAPS_PER_FILTER = IN_CHANNELS * TAPS_PER_CHANNEL      # 27

# Row-interleaved output: (row r, filter f) at element (r*OUT_CHANNELS + f)*OUT_COLS.
OUT_ROW_GROUP = OUT_CHANNELS * OUT_COLS  # 2048 elements: one output row, all 16 filters

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): every XMEM offset/base operand this app uses
# (ldr_mult_reg / ldr_cyclic_mult_reg's offset+base / str_post_aaq_reg) is a
# ROW number, not a byte address -- *_BASE_ADDR below stay byte constants used
# only to derive the ROW numbers; *_BASE_ROW feeds the CR registers the asm
# loads through. This app runs FP32 wide-vector only, so ROW_BYTES is always
# 512.

INPUT_BASE_ADDR = 0x000000    # 256 rows * 768 elements * 4 B
KERNEL_BASE_ADDR = 0x100000   # 16 filters x 128-element block * 4 B
MASK_BASE_ADDR = 0x104000     # 128-element mask blob (8 slots x 16 elements)
TEMP_BASE_ADDR = 0x104400     # 256 elements: temp0[0..127] (slot0 half) + temp1[128..255]
OUTPUT_BASE_ADDR = 0x140000   # 128 rows * 2048 elements * 4 B

CHUNK_ELEMENTS = 128
ROW_BYTES = CHUNK_ELEMENTS * 4  # 512 B/row in FP32 wide-vector mode

INPUT_BASE_ROW = INPUT_BASE_ADDR // ROW_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // ROW_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // ROW_BYTES
TEMP_BASE_ROW = TEMP_BASE_ADDR // ROW_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // ROW_BYTES

# Bias in element 0 of each filter's 128-element kernel block; 27 conv taps at [1..28).
BIAS_ELEMENT_OFFSET = 1
FILTER_BLOCK_ELEMENTS = 128

# -- Mask slots (polarity: bit 1 = KEEP, bit 0 = ZERO) -----------------------

MASK_SLOT_KEEP = 0    # all ones: interior taps, kc=0
MASK_SLOT_LEFT = 1    # zero lane 0:   kc=-1 on slot 0 (col 0 has no left nbr)
MASK_SLOT_RIGHT = 2   # zero lane 127: kc=+1 on slot 1 (col 255 has no right nbr)


def _build_mask_data() -> bytes:
    """Build the 128-byte mask blob (8 slots x 16 bytes, 128 bits each).

    cols per slot = 128 (one 128-lane slot of the 256-wide strip). The cyclic
    kc shift wraps the opposite image edge into the boundary lane, so:
      slot 0: all KEEP                 (interior / kc=0, and non-boundary slots)
      slot 1: ZERO lane 0              (kc=-1 applied to slot 0: strip position 0)
      slot 2: ZERO lane 127            (kc=+1 applied to slot 1: strip position 255)

    The mask blob does NOT widen with the active element width -- it is 1 bit
    per lane regardless of mode; only its row address (in ``setup()``) scales.
    """
    mask = bytearray(128)
    for slot in (MASK_SLOT_KEEP, MASK_SLOT_LEFT, MASK_SLOT_RIGHT):
        for bit in range(128):
            mask[slot * 16 + bit // 8] |= 1 << (bit % 8)
    mask[MASK_SLOT_LEFT * 16 + 0] &= ~(1 << 0)              # slot 1: zero lane 0
    mask[MASK_SLOT_RIGHT * 16 + 127 // 8] &= ~(1 << (127 % 8))  # slot 2: zero lane 127
    return bytes(mask)


class ConvFirstLayerApp(IpuApp):
    """First-layer 3x3 stride-2 conv + folded-bias + ReLU (256x256x3 -> 128x128x16, FP32).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to the assembled instruction binary.
        input_path:   Path to the input image binary, raw ``[3, 256, 256]``
                      float32.
        kernel:       Numpy weights of shape ``[16, 3, 3, 3]`` (out, in, kh, kw).
        kernel_path:  Alternative: raw ``[16, 3, 9]`` contiguous float32 file,
                      tap order kr=-1..+1, kc=-1..+1.
        bias:         Per-output-filter float32 bias, shape ``[16]``. Added
                      once to the accumulator before ReLU. Defaults to zeros.
        output_path:  Optional path to write the output (raw ``[16, 128,
                      128]`` float32).
    """

    ASM_PATH = Path(__file__).resolve().parent / "conv_first_layer.asm"

    def __init__(
        self,
        *,
        input_path: str | Path,
        kernel: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(input_path)

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if bias is None:
            bias = np.zeros(OUT_CHANNELS, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (OUT_CHANNELS,):
            raise ValueError(f"bias must have shape ({OUT_CHANNELS},), got {bias.shape}")
        self._bias_array = bias

    # -- kernel packing -------------------------------------------------

    def _kernel_taps(self) -> np.ndarray:
        """Return weights as ``[16, 3, 9]`` float32 (taps kr=-1..+1, kc=-1..+1)."""
        if self._kernel_array is not None:
            w = np.asarray(self._kernel_array, dtype=np.float32)
            if w.shape != (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE):
                raise ValueError(f"kernel must be [16, 3, 3, 3], got {w.shape}")
            return w.reshape(OUT_CHANNELS, IN_CHANNELS, 9)
        raw = self.kernel_path.read_bytes()
        expected = OUT_CHANNELS * IN_CHANNELS * 9 * 4
        if len(raw) != expected:
            raise ValueError(
                f"kernel_path file has {len(raw)} bytes, expected {expected}"
            )
        return np.frombuffer(raw, dtype=np.float32).reshape(OUT_CHANNELS, IN_CHANNELS, 9)

    def _build_kernel_data(self) -> bytes:
        """Pack 16 filters into 16 x 128-element blocks (float32).

        Each block: element 0 = filter bias, elements [1 .. 28) = 27 conv taps
        (ch0 taps 0..8, ch1 9..17, ch2 18..26), rest padding.
        """
        taps = self._kernel_taps()  # [16, 3, 9]
        packed = np.zeros(OUT_CHANNELS * FILTER_BLOCK_ELEMENTS, dtype=np.float32)
        for f in range(OUT_CHANNELS):
            dst = f * FILTER_BLOCK_ELEMENTS
            packed[dst] = self._bias_array[f]
            packed[dst + BIAS_ELEMENT_OFFSET:dst + BIAS_ELEMENT_OFFSET + TAPS_PER_FILTER] = (
                taps[f].reshape(-1)
            )
        return packed.tobytes()

    # -- state / setup ------------------------------------------------------

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
        # input_path holds the TRUE [in_channels, height, width] float32
        # tensor -- repack to row-interleaved-by-channel on-device layout.
        raw = np.frombuffer(self.input_path.read_bytes(), dtype=np.float32)
        expected = IN_CHANNELS * IN_ROWS * IN_COLS
        if raw.size != expected:
            raise ValueError(
                f"input has {raw.size} elements, expected {expected} "
                f"(3 ch * 256 rows * 256 cols)"
            )
        input_chw = raw.reshape(IN_CHANNELS, IN_ROWS, IN_COLS)
        input_data = np.ascontiguousarray(
            input_chw.transpose(1, 0, 2)  # [row, channel, col] -> row-interleaved
        )
        state.xmem.write_address(
            INPUT_BASE_ROW * ROW_BYTES, input_data.astype(np.float32).tobytes()
        )
        state.xmem.write_address(
            KERNEL_BASE_ROW * ROW_BYTES, self._build_kernel_data(),
        )
        # The mask blob does NOT widen (1 bit/lane regardless of mode) -- only
        # its row address scales.
        state.xmem.write_address(MASK_BASE_ROW * ROW_BYTES, _build_mask_data())

        # CR15 dstructure: one 128-lane slot is fully valid.
        state.set_cr_dstructure(valid_elements=128)

        # CR map (ROW-space; see the module-level note on the address-space
        # split). CR0 == 0, CR1 == 1 are read-only constants; CR15 is dstructure.
        state.regfile.set_cr(3, KERNEL_BASE_ROW)
        state.regfile.set_cr(4, MASK_BASE_ROW)
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, CHANNEL_STRIDE // CHUNK_ELEMENTS)     # 2: ch-to-ch step, rows
        state.regfile.set_cr(7, IN_ROW_GROUP // CHUNK_ELEMENTS)       # 6: one input spatial row (3 ch), rows
        state.regfile.set_cr(8, FILTER_BLOCK_ELEMENTS // CHUNK_ELEMENTS)  # 1: chunk / slot size, rows
        state.regfile.set_cr(9, 2 * IN_ROW_GROUP // CHUNK_ELEMENTS)   # 12: input advance per OUTPUT row (stride 2), rows
        state.regfile.set_cr(10, INPUT_BASE_ROW)                  # input base row
        state.regfile.set_cr(11, OUT_ROWS)                        # 128: output-row loop bound (not XMEM-space)
        state.regfile.set_cr(12, OUT_CHANNELS * FILTER_BLOCK_ELEMENTS // CHUNK_ELEMENTS)  # 16: kernel limit / output row group, rows
        state.regfile.set_cr(13, TEMP_BASE_ROW)                   # temp0 (slot0 half) at +0, temp1 at +128
        # cr14 = 128: r_cyclic ELEMENT slot1 index, split off cr8 (which is now
        # the XMEM filter-block ROW stride = 1 and would otherwise collide).
        state.regfile.set_cr(14, 128)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_elements = OUT_ROWS * OUT_ROW_GROUP  # 128 * 2048 = 262144
            raw = state.xmem.read_address(
                OUTPUT_BASE_ROW * ROW_BYTES, total_elements * 4
            )
            out = np.frombuffer(raw, dtype=np.float32).reshape(OUT_ROWS, OUT_CHANNELS, OUT_COLS)
            out_chw = np.ascontiguousarray(out.transpose(1, 0, 2))  # -> [filter, row, col]
            Path(self.output_path).write_bytes(out_chw.astype(np.float32).tobytes())

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)


# -- registry declaration ---------------------------------------------------
# Fixed shape: 256x256x3 -> 128x128x16, stride 2, padding 1, kernel_size 3.
# bias is always folded and ReLU always applied.


def _supports(**params):
    q = conv_query(**params)
    if bad := positive_dims(q):
        return no(bad)
    if q.kernel_size != 3:
        return no(f"handles only kernel_size=3; got {q.kernel_size}")
    if q.dilation != 1:
        return no(f"handles only dilation=1; got {q.dilation}")
    if q.padding != 1:
        return no(f"handles only padding=1; got {q.padding}")
    if q.stride != 2:
        return no(f"handles only stride=2; got {q.stride}")
    if q.groups != 1:
        return no(f"handles only groups=1 (plain conv); got {q.groups}")
    if not q.apply_relu:
        return no("this kernel always applies ReLU")
    if not q.has_bias:
        return no("this kernel requires bias (folded)")
    if (q.in_channels, q.out_channels, q.height, q.width) != (IN_CHANNELS, OUT_CHANNELS, IN_ROWS, IN_COLS):
        return no(
            f"fixed-shape kernel: only in_channels={IN_CHANNELS}, "
            f"out_channels={OUT_CHANNELS}, height={IN_ROWS}, width={IN_COLS} "
            f"are supported; got in_channels={q.in_channels}, "
            f"out_channels={q.out_channels}, height={q.height}, width={q.width}"
        )
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"kernel_size=3, stride=2, padding=1, bias+ReLU, fixed shape "
        f"{IN_CHANNELS}x{IN_ROWS}x{IN_COLS} -> {OUT_CHANNELS}x{OUT_ROWS}x{OUT_COLS}: "
        f"the network's first-layer conv (FP32), hand-optimized for this exact shape."
    )


def _caveats(**params):
    return (
        "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
        "kernel has no INT8/quantized variant.",
        f"Fixed shape only: {IN_CHANNELS}x{IN_ROWS}x{IN_COLS} -> "
        f"{OUT_CHANNELS}x{OUT_ROWS}x{OUT_COLS}. No parameterization.",
    )


SPEC = KernelSpec(
    name="conv_first_layer",
    op="conv2d",
    variant="first_layer",
    app_class=ConvFirstLayerApp,
    asm="conv_first_layer.asm",
    requires=REQUIRES,
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=_caveats,
    bundle=lambda **params: conv_query(**params).bundle,
    cost=lambda **params: 0.0,
)
