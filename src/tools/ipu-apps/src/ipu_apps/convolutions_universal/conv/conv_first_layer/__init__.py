"""First-layer 3x3 convolution + folded-bias + ReLU: 256x256x3 -> 128x128x16.

Hardcoded shapes for the network's first layer (stride 2, pad 1):

    Input:  256x256x3   (INT8, ROW-INTERLEAVED by channel: for each spatial row
                         r, ch0's 256 cols, then ch1's, then ch2's, then row r+1.
                         Address of (row r, channel ch): (r*3 + ch)*256.)
    Kernel: 16 filters x 3 channels x 3x3 taps, packed by the harness.
    Output: 128x128x16  (INT8, ROW-INTERLEAVED by channel, written to its FINAL
                         address by the asm: (row r, filter f) -> (r*16 + f)*128.
                         One output row = one 128-byte chunk (never split).)

The **input image is stored verbatim** — the harness only places the raw bytes in
the row-interleaved order above; it does NOT slice, pad, or seam-fix. All conv
logic (windowing, borders, stride, the half seam) lives in the asm.

This is the ``_bn_activation`` flavour: per-output-filter **BN bias** is folded
into the kernel (byte 0 of each filter's 128-byte block) and accumulated once
before the conv taps; **ReLU** is applied before INT8 quantization. Batch-norm
scale is assumed folded into the conv weights, so only the bias remains separate.

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
        input_path="input.bin",   # 256*256*3 bytes, row-interleaved, INT8
        kernel=weights,           # np.ndarray [16, 3, 3, 3]  (out,in,kh,kw)
        bias=bias_int8,           # np.ndarray [16], folded BN bias (INT8)
        output_path="output.bin",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal.weights import cast_to_wire_bytes

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Fixed shapes ------------------------------------------------------------

IN_ROWS = 256
IN_COLS = 256
IN_CHANNELS = 3
OUT_ROWS = 128
OUT_COLS = 128
OUT_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 2

# Row-interleaved input: (row r, channel ch) at (r*IN_CHANNELS + ch)*IN_COLS.
IN_ROW_GROUP = IN_CHANNELS * IN_COLS   # 768 bytes: one spatial row, all 3 channels
CHANNEL_STRIDE = IN_COLS               # 256: ch-to-ch step within a row group

TAPS_PER_CHANNEL = KERNEL_SIZE * KERNEL_SIZE          # 9
TAPS_PER_FILTER = IN_CHANNELS * TAPS_PER_CHANNEL      # 27

# Row-interleaved output: (row r, filter f) at (r*OUT_CHANNELS + f)*OUT_COLS.
OUT_ROW_GROUP = OUT_CHANNELS * OUT_COLS  # 2048 bytes: one output row, all 16 filters

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): every XMEM offset/base operand this app uses
# (ldr_mult_reg / ldr_cyclic_mult_reg's offset+base / str_post_aaq_reg) is a
# ROW number, not a byte address -- *_BASE_ADDR below stay narrow-byte
# constants used only to derive the ROW numbers; *_BASE_ROW feeds the CR
# registers the asm loads through, and (scaled by the active mode's row size)
# the host-side xmem pokes in setup()/teardown(). Unlike the depthwise/conv
# apps, this app has NO r_cyclic-index CR overload to split: every
# ldr_cyclic_mult_reg's `index` operand here is a literal LR holding 0 or 128
# (lr0/lr11), never a CR, so all the byte-stride CRs (256/768/128/1536/2048)
# are purely XMEM-space and divide evenly by 128.
#
# Mode-blindness: host-side pokes (write_address/read_address) are always
# byte-granular and mode-UNAWARE, so setup()/teardown() must scale every
# address by the active mode's row size (128 B narrow, 512 B wide-vector
# debug) themselves -- see ``self._element_width`` / ``row_bytes`` below.
# Packed data must also widen from 1 B/element to 4 B/element so each row's
# 128 elements still fill exactly one row's worth of bytes in wide mode
# (128*4 = 512, no gap); see ``_pack_kernel_data``/``_pack_mask_data``.

INPUT_BASE_ADDR = 0x000000    # 256 rows * 768 B = 196608 bytes
KERNEL_BASE_ADDR = 0x040000   # 16 filters x 128-byte block = 2048 bytes
MASK_BASE_ADDR = 0x041000     # 128-byte mask blob (8 slots x 16 bytes)
TEMP_BASE_ADDR = 0x041100     # 256 bytes: temp0[0..127] (slot0 half) + temp1[128..255]
OUTPUT_BASE_ADDR = 0x050000   # 128 rows * 2048 B = 262144 bytes

CHUNK_BYTES = 128
INPUT_BASE_ROW = INPUT_BASE_ADDR // CHUNK_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // CHUNK_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // CHUNK_BYTES
TEMP_BASE_ROW = TEMP_BASE_ADDR // CHUNK_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES

# Bias in byte 0 of each filter's 128-byte kernel block; 27 conv taps at [1..28).
BIAS_BYTE_OFFSET = 1
FILTER_BLOCK_BYTES = 128

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

    The mask blob does NOT widen with the active element width -- like every
    other app in this package, it is 1 bit per lane in both modes; only its
    row address (in ``setup()``) scales.
    """
    mask = bytearray(128)
    for slot in (MASK_SLOT_KEEP, MASK_SLOT_LEFT, MASK_SLOT_RIGHT):
        for bit in range(128):
            mask[slot * 16 + bit // 8] |= 1 << (bit % 8)
    mask[MASK_SLOT_LEFT * 16 + 0] &= ~(1 << 0)              # slot 1: zero lane 0
    mask[MASK_SLOT_RIGHT * 16 + 127 // 8] &= ~(1 << (127 % 8))  # slot 2: zero lane 127
    return bytes(mask)


def _as_signed_byte(value: int) -> int:
    """Reinterpret a wire byte as the signed INT8 it encodes."""
    v = value & 0xFF
    return v - 256 if v > 127 else v


def _widen_int8_bytes(data: bytes, element_width: int) -> bytes:
    """Widen a 1 B/element INT8 byte string to ``element_width`` B/element.

    Every element becomes a little-endian signed 32-bit int at ``element_width
    == 4`` (wide-vector debug mode), so a 128-byte narrow row still occupies
    exactly one row's worth of bytes (128*4 = 512) with no gap. A no-op at
    ``element_width == 1`` (narrow mode).
    """
    if element_width == 1:
        return data
    out = bytearray(len(data) * element_width)
    for i, b in enumerate(data):
        struct.pack_into("<i", out, i * element_width, _as_signed_byte(b))
    return bytes(out)


class ConvFirstLayerApp(IpuApp):
    """First-layer 3x3 stride-2 conv + folded-bias + ReLU (256x256x3 -> 128x128x16).

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to the assembled instruction binary.
        input_path:   Path to the input image binary (256*256*3 bytes,
                      row-interleaved by channel, INT8).
        kernel:       Numpy weights of shape ``[16, 3, 3, 3]`` (out, in, kh, kw).
        kernel_path:  Alternative: raw ``[16, 3, 9]`` contiguous int8 file
                      (432 bytes), tap order kr=-1..+1, kc=-1..+1.
        bias:         Per-output-filter INT8 bias, shape ``[16]``. Added once to
                      the accumulator before ReLU. Defaults to zeros.
        output_path:  Optional path to write the output region verbatim
                      (row-interleaved 128x128x16, ready for the next layer).
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
        self.dtype = DType.INT8

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        if bias is None:
            bias = np.zeros(OUT_CHANNELS, dtype=np.int8)
        bias = np.asarray(bias)
        if bias.shape != (OUT_CHANNELS,):
            raise ValueError(f"bias must have shape ({OUT_CHANNELS},), got {bias.shape}")
        self._bias_array = bias

    # -- kernel packing (input image is left untouched) ---------------------

    def _kernel_taps(self) -> np.ndarray:
        """Return weights as ``[16, 3, 9]`` int8 (taps kr=-1..+1, kc=-1..+1)."""
        if self._kernel_array is not None:
            w = np.asarray(self._kernel_array)
            if w.shape != (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE):
                raise ValueError(f"kernel must be [16, 3, 3, 3], got {w.shape}")
            return w.reshape(OUT_CHANNELS, IN_CHANNELS, 9)
        raw = self.kernel_path.read_bytes()
        expected = OUT_CHANNELS * IN_CHANNELS * 9
        if len(raw) != expected:
            raise ValueError(
                f"kernel_path file has {len(raw)} bytes, expected {expected}"
            )
        return np.frombuffer(raw, dtype=np.int8).reshape(OUT_CHANNELS, IN_CHANNELS, 9)

    def _build_kernel_data(self) -> bytes:
        """Pack 16 filters into 16 x 128-byte blocks (1 B/element, narrow-mode
        layout -- widened to the active element width by ``setup()``).

        Each block: byte 0 = filter bias, bytes [1 .. 28) = 27 conv taps
        (ch0 taps 0..8, ch1 9..17, ch2 18..26), rest padding.
        """
        taps = self._kernel_taps()                        # [16, 3, 9]
        tap_bytes = cast_to_wire_bytes(taps, self.dtype)  # 16*27 contiguous
        bias_bytes = cast_to_wire_bytes(self._bias_array, self.dtype)
        block = bytearray(OUT_CHANNELS * FILTER_BLOCK_BYTES)
        for f in range(OUT_CHANNELS):
            dst = f * FILTER_BLOCK_BYTES
            block[dst] = bias_bytes[f]
            src = f * TAPS_PER_FILTER
            block[dst + BIAS_BYTE_OFFSET:dst + BIAS_BYTE_OFFSET + TAPS_PER_FILTER] = (
                tap_bytes[src:src + TAPS_PER_FILTER]
            )
        return bytes(block)

    # -- state / setup ------------------------------------------------------

    @staticmethod
    def make_state() -> "IpuState":
        from ipu_emu.ipu_state import IpuState
        return IpuState()

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype

        # Element width of the active mode: 1 B narrow, 4 B wide-vector debug.
        # Row *numbers* handed to CRs are mode-independent, but the host-side
        # byte pokes below must land at the same rows, so both the row stride
        # and the packed payloads (1 B/element on disk) scale by it.
        element_width = 4 if state.wide_vector_debug else 1
        row_bytes = CHUNK_BYTES * element_width

        input_data = self.input_path.read_bytes()
        if len(input_data) != IN_ROWS * IN_ROW_GROUP:
            raise ValueError(
                f"input has {len(input_data)} bytes, expected "
                f"{IN_ROWS * IN_ROW_GROUP} (256 rows * 3 ch * 256 cols)"
            )
        state.xmem.write_address(
            INPUT_BASE_ROW * row_bytes, _widen_int8_bytes(input_data, element_width)
        )
        state.xmem.write_address(
            KERNEL_BASE_ROW * row_bytes,
            _widen_int8_bytes(self._build_kernel_data(), element_width),
        )
        # The mask blob does NOT widen (1 bit/lane in both modes) -- only its
        # row address scales.
        state.xmem.write_address(MASK_BASE_ROW * row_bytes, _build_mask_data())

        # CR15 dstructure: one 128-lane slot is fully valid.
        state.set_cr_dstructure(valid_elements=128)

        # CR map (ROW-space; see the module-level note on the address-space
        # split). CR0 == 0, CR1 == 1 are read-only constants; CR15 is dstructure.
        state.regfile.set_cr(3, KERNEL_BASE_ROW)
        state.regfile.set_cr(4, MASK_BASE_ROW)
        state.regfile.set_cr(5, OUTPUT_BASE_ROW)
        state.regfile.set_cr(6, CHANNEL_STRIDE // CHUNK_BYTES)     # 2: ch-to-ch step, rows
        state.regfile.set_cr(7, IN_ROW_GROUP // CHUNK_BYTES)       # 6: one input spatial row (3 ch), rows
        state.regfile.set_cr(8, FILTER_BLOCK_BYTES // CHUNK_BYTES)  # 1: chunk / slot size, rows
        state.regfile.set_cr(9, 2 * IN_ROW_GROUP // CHUNK_BYTES)   # 12: input advance per OUTPUT row (stride 2), rows
        state.regfile.set_cr(10, INPUT_BASE_ROW)                  # input base row
        state.regfile.set_cr(11, OUT_ROWS)                        # 128: output-row loop bound (not XMEM-space)
        state.regfile.set_cr(12, OUT_CHANNELS * FILTER_BLOCK_BYTES // CHUNK_BYTES)  # 16: kernel limit / output row group, rows
        state.regfile.set_cr(13, TEMP_BASE_ROW)                   # temp0 (slot0 half) at +0, temp1 at +128
        # cr14 = 128: r_cyclic ELEMENT slot1 index, split off cr8 (which is now
        # the XMEM filter-block ROW stride = 1 and would otherwise collide).
        state.regfile.set_cr(14, 128)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # str_post_aaq_reg always writes a fixed 512-byte POST_AAQ_REG
            # payload per output row (see ipu.py::execute_str_post_aaq_reg),
            # which spans 4 rows in narrow mode (128 B each, INT8-packed) but
            # only 1 row in wide-vector debug mode (512 B, 4 B/element). Read
            # back at the mode's actual row size, then narrow to 1 B/element
            # so ``output_path`` always holds the same canonical INT8 layout.
            element_width = 4 if state.wide_vector_debug else 1
            row_bytes = CHUNK_BYTES * element_width
            total_elements = OUT_ROWS * OUT_ROW_GROUP  # 128 * 2048 = 262144
            raw = state.xmem.read_address(
                OUTPUT_BASE_ROW * row_bytes, total_elements * element_width
            )
            if element_width == 1:
                data = bytes(raw)
            else:
                data = bytes(
                    _as_signed_byte(struct.unpack_from("<i", raw, i * 4)[0]) & 0xFF
                    for i in range(total_elements)
                )
            Path(self.output_path).write_bytes(data)

    def run(self, **kwargs):
        kwargs.setdefault("state", self.make_state())
        return super().run(**kwargs)
