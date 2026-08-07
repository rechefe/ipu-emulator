"""Universal standard 3x3 convolution harness.

A single parameterized harness that works for ANY valid standard convolution
configuration (spatial >= 16x16, in_channels >= 1). Same chunk-interleaved
I/O layout and FPB=28 super-block kernel packing, no bias/activation folding
(see ``conv_universal_bn_activation`` for the bias+ReLU twin).

Pipeline per filter: r_acc seeded from the first 3x3 conv tap
(``ACC.ADD.FIRST``), then += the remaining taps over all input channels, then
``ACTIVATE.QUANTIZE identity`` -> store 128 B.

Kernel super-block layout (FPB=28):
  One 256-byte super-block holds up to 28 input-channel slots of one output
  filter. Channel ``s`` occupies bytes ``[s*9 .. s*9 + 9)``: 28 * 9 = 252
  bytes <= 256. Channels 0..13 land in the first 128-byte half (R0), 14..27
  in the second (R1); the shared-index ``mult.ve`` (fixed_idx 0..255)
  addresses all 28.

Usage::

    from ipu_apps.convolutions_universal.conv.conv_universal import ConvUniversalApp

    # Numpy weights form (preferred):
    app = ConvUniversalApp(
        inst_path="conv_universal.bin",
        input_path="input.bin",
        kernel=weights_nhwc,      # np.ndarray [out_ch, in_ch, 3, 3]
        output_path="output.bin",
        dtype="INT8",
        rows=32, cols=32, in_channels=16, out_channels=16,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    CHUNK_BYTES,
    parse_dtype,
    dump_outputs,
)
from ipu_apps.convolutions_universal.weights import cast_to_wire_bytes

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------

# XMEM byte addresses. These are used for *host-side* pokes
# (``state.xmem.write_address`` / ``dump_outputs``), which are direct memory
# accesses and always byte-granular. The ISA-visible forms handed to CRs are
# the ``*_ROW`` values below -- see "Row addressing" note further down.
INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x100000
# Border handling is done entirely with masks (no zero-region).  A SINGLE
# 128-byte R_MASK blob carries all 3 slots (0=none, 3=top-row zero, 6=bottom-row
# zero); it is loaded once at init.  The vertical out-of-bounds row is zeroed by
# a mask slot instead of loading a zero chunk into the cyclic register; left/
# right edge columns are applied at runtime by mask_shift.
MASK_BASE_ADDR = 0x180000     # single mask blob (slots 0/3/6)
OUTPUT_BASE_ADDR = 0x1C0000

# -- Row addressing ----------------------------------------------------------
# XMEM instruction operands (``offset + base`` on LDR_*/store) are ROW numbers,
# not byte addresses: the emulator scales them by the active mode's row size
# (128 B narrow, 512 B wide-vector debug). Keeping every ISA-visible address in
# rows is what lets one .asm run unchanged in both modes -- the same row number
# names the same logical row regardless of element width.
#
# Only *XMEM-space* quantities convert. r_cyclic operands (MULT.RC.* ``rc_idx``
# reads and LDR_CYCLIC_MULT_REG's ``index`` writes) are ELEMENT-indexed and the
# ring is 512 elements in both modes, so they are already mode-blind and must
# NOT be rescaled. See ipu.py::_xmem_row_addr and _rc_element_to_byte_offset.
INPUT_BASE_ROW = INPUT_BASE_ADDR // CHUNK_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // CHUNK_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // CHUNK_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES

OUTPUT_CHUNK_BYTES = CHUNK_BYTES  # bytes per output filter per chunk (int8)

SUPER_BLOCK_BYTES = 2 * CHUNK_BYTES  # 256 = R0 half + R1 half
SUPER_BLOCK_ROWS = 2                 # the same super-block, in XMEM rows
HALF_FPB = CHUNK_BYTES // 9          # 14: channels per 128-byte half (9 taps each)
FPB = 2 * HALF_FPB                   # 28: channels per super-block (R0+R1 shared index)


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
    ``[r*cols, r*cols + cols)``.

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


def _as_signed_byte(value: int) -> int:
    """Reinterpret a wire byte as the signed INT8 it encodes."""
    v = value & 0xFF
    return v - 256 if v > 127 else v


def _pack_conv_weights_fpb28(
    weights_reordered: np.ndarray, dtype: DType, element_width: int = 1
) -> bytes:
    """Pack [out_ch, in_ch, 9] (taps already reordered) into FPB=28 super-blocks.

    Each super-block is 256 ELEMENTS laid out linearly: channel ``s`` occupies
    elements ``[s*9 .. s*9+9)``. The first 128 elements are loaded into R0 and
    the second 128 into R1; the asm uses mult.ve with a shared fixed_idx
    (0..255) that sweeps the entire super-block.

    The layout is ELEMENT-identical in both modes; only the byte scale differs
    (1 B/element narrow, 4 B/element wide-vector debug) since MULT.VE indexes
    ra by lane in wide mode and by byte in narrow. Per-filter row stride:
    ceil(in_ch / 28) * SUPER_BLOCK_ROWS.
    """
    out_ch, in_ch, k2 = weights_reordered.shape
    if k2 != 9:
        raise ValueError(f"expected last dim=9 (taps), got {k2}")
    raw = cast_to_wire_bytes(weights_reordered, dtype)
    # raw indexing: byte for filter f, channel ic, tap t = raw[(f*in_ch+ic)*9 + t]

    super_blocks_per_filter = math.ceil(in_ch / FPB)
    total = out_ch * super_blocks_per_filter * SUPER_BLOCK_BYTES * element_width
    packed = bytearray(total)

    def put(elem_idx: int, value: int) -> None:
        if element_width == 1:
            packed[elem_idx] = value & 0xFF
        else:
            struct.pack_into("<i", packed, elem_idx * 4, _as_signed_byte(value))

    for f in range(out_ch):
        for sb in range(super_blocks_per_filter):
            sb_base = (f * super_blocks_per_filter + sb) * SUPER_BLOCK_BYTES
            for s in range(FPB):
                ic = sb * FPB + s
                if ic >= in_ch:
                    break
                src = (f * in_ch + ic) * 9
                dst = sb_base + s * 9   # linear layout: 0,9,18,...,243
                for t in range(9):
                    put(dst + t, raw[src + t])
    return bytes(packed)


class ConvUniversalApp(IpuApp):
    """Universal standard 3x3 convolution application harness.

    Exactly one of ``kernel`` or ``kernel_path`` must be supplied.

    Args:
        inst_path:    Path to assembled universal binary.
        input_path:   Path to input image binary (chunk-interleaved layout).
        kernel:       Numpy weights of shape ``[out_ch, in_ch, 3, 3]``.
                      Packed at setup via :func:`pack_conv_weights_dense`.
        kernel_path:  Alternative: path to a raw ``[out_ch, in_ch, 9]``
                      contiguous byte file. Reshaped to ``[out_ch, in_ch, 3, 3]``
                      and packed the same way.
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width; one of {16, 32, 64, 128} (one packed row
                      per mask partition group; cols=128 uses Partition.P0).
        in_channels:  Number of input channels (>= 1).
        out_channels: Number of output channels (>= 1).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        in_channels: int,
        out_channels: int,
        kernel: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        kernel_path = getattr(self, "kernel_path", None)
        if kernel is not None and kernel_path is not None:
            raise ValueError("Provide exactly one of kernel= or kernel_path=")
        if kernel is None and kernel_path is None:
            raise ValueError("Provide one of kernel= or kernel_path=")
        self._kernel_array = kernel
        self.kernel_path = Path(kernel_path) if kernel_path is not None else None

        # Validate
        valid_cols = {16, 32, 64, 128}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if in_channels < 1:
            raise ValueError(f"in_channels ({in_channels}) must be >= 1")
        if out_channels < 1:
            raise ValueError(f"out_channels ({out_channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_chunks = num_chunks
        # Row-granular strides (see "Row addressing" above). One chunk is one
        # XMEM row, so a group of ``in_channels`` chunks is ``in_channels`` rows.
        self.in_group_stride = in_channels
        self.blocks_per_filter = math.ceil(in_channels / FPB)
        # A super-block spans SUPER_BLOCK_ROWS (= 2) rows: the R0 half + R1 half.
        self.total_kernel_rows = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_ROWS
        )
        self.total_kernel_bytes = (
            out_channels * self.blocks_per_filter * SUPER_BLOCK_BYTES
        )
        # Input sits one group stride above cr_input_base so the g0 kr=-1
        # prefetch (offset lr_chunk_base - cr6) bottoms out at exactly 0.
        self.input_data_row = INPUT_BASE_ROW + self.in_group_stride
        # Narrow-mode default; setup() overrides it once the state's mode is known.
        self._element_width = 1

    def _pack_kernel(self) -> bytes:
        if self._kernel_array is not None:
            weights = self._kernel_array
        else:
            raw = self.kernel_path.read_bytes()
            expected = self.out_channels * self.in_channels * 9
            if len(raw) != expected:
                raise ValueError(
                    f"kernel_path file has {len(raw)} bytes, "
                    f"expected {expected} (out_ch * in_ch * 9)"
                )
            # Raw bytes are assumed to be int8 for INT8 dtype
            # (FP8 rawbytes must be supplied as a kernel= numpy float32 array).
            if self.dtype != DType.INT8:
                raise ValueError(
                    "kernel_path is only supported for INT8; use kernel= "
                    "for FP8 dtypes"
                )
            weights = (
                np.frombuffer(raw, dtype=np.int8)
                .reshape(self.out_channels, self.in_channels, 3, 3)
            )
        # Tap order in the walking-pointer asm: kr=-1 → kr=0 → kr=+1, with
        # kc=-1 → 0 → +1 within each row.  That's natural row-major from the
        # source [out_ch, in_ch, 3, 3] — no reordering needed.
        w_reordered = weights.reshape(self.out_channels, self.in_channels, 9)
        return _pack_conv_weights_fpb28(w_reordered, self.dtype, self._element_width)

    def setup(self, state: "IpuState") -> None:
        # Set data type (master ISA: dtype is a state attribute, not a CR register).
        state.dtype = self.dtype

        # Element width of the active mode: 1 B narrow, 4 B wide-vector debug.
        # Row *numbers* handed to CRs are mode-independent, but the host-side
        # byte pokes below must land at the same rows, so they scale by it.
        self._element_width = 4 if state.wide_vector_debug else 1
        row_bytes = CHUNK_BYTES * self._element_width

        # Load input
        input_data = self.input_path.read_bytes()
        state.xmem.write_address(self.input_data_row * row_bytes, input_data)

        # Pack and load kernel (dense FPB=28 super-block layout)
        kernel_packed = self._pack_kernel()
        state.xmem.write_address(KERNEL_BASE_ROW * row_bytes, kernel_packed)

        # Border masks: a SINGLE blob carrying all 3 slots (0=none, 3=top-row
        # zero, 6=bottom-row zero), loaded once at init.  The g0 section selects
        # slot 3, the gN section selects slot 6 — no mid-program R_MASK reload.
        # Vertical out-of-bounds rows are masked (no zero chunk in the cyclic
        # register); left/right edge columns are applied at runtime by mask_shift
        # (see CR15 partition below). The mask blob does NOT widen -- it is 1
        # bit per lane in both modes -- only its row address scales.
        state.xmem.write_address(
            MASK_BASE_ROW * row_bytes, build_border_mask_blob(self.cols)
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
        if self.cols not in cols_to_partition:
            raise ValueError(
                f"conv_universal mask-shift scheme requires cols in "
                f"{sorted(cols_to_partition)} (one packed row per partition group); "
                f"got cols={self.cols}"
            )
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
        # All four are XMEM *row* numbers, not byte addresses (see "Row
        # addressing" above) -- the emulator scales them per active mode.
        state.regfile.set_cr(10, INPUT_BASE_ROW)
        state.regfile.set_cr(5, KERNEL_BASE_ROW)
        state.regfile.set_cr(2, OUTPUT_BASE_ROW)
        state.regfile.set_cr(3, MASK_BASE_ROW)           # single mask blob (slots 0/3/6)

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
        # advance 384). The ring is 512 ELEMENTS in both modes, so these are
        # already mode-blind and must NOT be divided by CHUNK_BYTES.
        #
        # Their former *XMEM* role (advance one chunk = one row) is now served
        # by the read-only constant CR1 (= 1 row), since in row space a chunk
        # stride is literally 1. That split is why cr12 is no longer overloaded.
        state.regfile.set_cr(12, CHUNK_BYTES)        # 128 ELEMENTS (r_cyclic slot stride)
        state.regfile.set_cr(13, SUPER_BLOCK_ROWS)   # 2 ROWS (kernel super-block stride)
        # cr9 = ring advance = 3 * 128 = 384 ELEMENTS.  9-cyc role-rotating scheme:
        # lr_read (kr=0 slot) advances -128 (= +384 mod 512) per channel.
        state.regfile.set_cr(9, 3 * CHUNK_BYTES)     # 384 ELEMENTS
        # cr14 = end-of-9 walking-pointer wrap step: brings lr_walk from this ch's
        # tap-9 offset (lr_read + cols + 1) to next ch's tap-1 offset
        # ((lr_read - CHUNK_BYTES) - cols - 1).  Under -128 rotation this is
        # +(RING_ADV - 2*cols - 2) with RING_ADV = 384 (= -128 mod 512 + 512).
        state.regfile.set_cr(14, (3 * CHUNK_BYTES - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.out_channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
