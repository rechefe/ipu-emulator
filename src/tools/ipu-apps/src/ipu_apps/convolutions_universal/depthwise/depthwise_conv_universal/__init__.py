"""Universal depthwise 3x3 convolution harness (no bias, no activation).

Base app for ``depthwise_conv_universal_bn_activation``. Same chunk-interleaved
I/O layout and walking-pointer / rotating-cyclic-slot pipeline, minus:

  * **no folded bias** — ``r_acc`` is seeded by tap 1's own product
    (``acc.add.first``) instead of a bias multiply-broadcast.
  * **no ReLU** — ``ACTIVATE.QUANTIZE`` runs with ``identity``; it is still
    required as the only INT8 clamp/quantize step (``r_acc`` is not directly
    storable).

Per-channel budget: **10 cyc/ch** = 9 weight taps + 1 ACTIVATE cycle (tap 1
doubles as the r_acc reset via ``acc.add.first``, replacing the BN twin's
separate bias-seed cycle — the one cycle actually saved). ACTIVATE still
needs its own cycle (reads the cycle-start snapshot of ``r_acc``, same as the
BN twin's placeholder cycle) and now also co-issues the next channel's kr=-1
prefetch load, since ACTIVATE occupies its own slot type and leaves the
LR/XMEM slots free that cycle — replacing the load the BN twin's bias-seed
cycle used to carry. The deferred-store pipeline (store the previous
channel's quantized result while the current channel computes) is preserved.

Kernel super-block layout (FPB=28, 9-byte stride, no bias byte):
  Depthwise produces one output PER channel; each channel occupies a **9-byte
  slot** (its 9 weight taps only — no bias byte to reserve). 28 channels * 9 =
  252 <= 256, so one 256-byte super-block (R0 = bytes 0..127, R1 = 128..255)
  holds 28 channels. The shared ``mult.ve`` fixed_idx (0..255) addresses both
  halves transparently.

  The asm walks one continuous kernel byte index ``lr6`` at +1 per cycle: for
  channel ``s`` taps 1..9 read ``fixed_idx = s*9 .. s*9 + 9``; the next
  channel's tap 1 is the following byte, so the 9-cycle/channel body advances
  ``lr6`` by exactly one channel stride.

Usage::

    from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
        DepthwiseConvUniversalApp,
    )

    app = DepthwiseConvUniversalApp(
        inst_path="depthwise_conv_universal.bin",
        input_path="input.bin",
        kernel_path="kernel.bin",   # channels * 9 bytes, INT8 raw
        output_path="output.bin",
        dtype="INT8",
        rows=64, cols=64, channels=256,
    )
    state, cycles = app.run()
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import Partition

from ipu_apps.base import IpuApp
from ipu_apps.convolutions_universal import (
    parse_dtype,
    dump_outputs,
)
# Reuse the conv_universal_bn_activation mask-blob builder so the depthwise
# apps share one border-mask implementation: a single 128-byte blob (slots
# 0/3/6) where left/right edge columns are applied at runtime via mask_shift
# (CR15 partition).
from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    build_border_mask_blob,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Memory layout -----------------------------------------------------------
#
# Row-addressed ISA (mb/195): XMEM offset/base operands on LDR_*/STR_*
# (including LDR_CYCLIC_MULT_REG's offset/base -- only its `index` is
# r_cyclic-element-space) are ROW numbers, not byte addresses. *_BASE_ADDR
# below stay as byte constants for host-side xmem pokes (write_address /
# read_address are byte-granular); *_BASE_ROW = *_BASE_ADDR // CHUNK_BYTES
# feeds the CR registers the asm actually loads/stores through.
#
# r_cyclic ELEMENT addressing is untouched by this migration: lr5 (the
# LDR_CYCLIC_MULT_REG `index`) and lr3/lr4 (MULT.RC.VE `rc_idx` / read-slot
# rotation) index a 512-ELEMENT ring in both modes, so cr12/cr13/cr9/cr14 in
# their r_cyclic role (slot-size 128, slot-step 256/384) are unchanged.

INPUT_BASE_ADDR = 0x000000
KERNEL_BASE_ADDR = 0x110000
# Border handling is done entirely with masks (no zero region). A single
# 128-byte blob carries all three slots (0=none, 3=top-row zero, 6=bottom-row
# zero); the g0 section selects slot 3 and the gN section selects slot 6, so no
# mid-program R_MASK reload is needed.  Left/right edge columns are applied at
# runtime by mask_shift (CR15 partition = cols).
MASK_BASE_ADDR = 0x120000
OUTPUT_BASE_ADDR = 0x130000

OUTPUT_CHUNK_BYTES = 128  # 128 bytes per output channel per chunk (int8)
CHUNK_BYTES = 128         # XMEM row size in narrow mode; row-number unit

INPUT_BASE_ROW = INPUT_BASE_ADDR // CHUNK_BYTES
KERNEL_BASE_ROW = KERNEL_BASE_ADDR // CHUNK_BYTES
MASK_BASE_ROW = MASK_BASE_ADDR // CHUNK_BYTES
OUTPUT_BASE_ROW = OUTPUT_BASE_ADDR // CHUNK_BYTES

FPB = 28           # channels per 256-byte super-block (9 taps each, no bias)
CH_SLOT_BYTES = 9  # per-channel slot: 9 weight taps, no bias byte
SUPER_BLOCK_BYTES = 256
SUPER_BLOCK_ROWS = SUPER_BLOCK_BYTES // CHUNK_BYTES


def _as_signed_byte(value: int) -> int:
    v = value & 0xFF
    return v - 256 if v > 127 else v


def _pack_depthwise_kernel(
    kernel_raw: bytes, channels: int, element_width: int = 1,
) -> bytes:
    """Pack per-channel 9 weight taps into FPB=28 super-blocks.

    Input:  ``kernel_raw`` = channels*9 bytes (channel ch's 9 taps at ch*9).
    Output: ceil(channels/28) super-blocks of ``256 * element_width`` bytes.

    Within one super-block, channel ``s`` (0..27) occupies ELEMENTS
    ``[s*9 .. s*9 + 9)``. 28*9 = 252 <= 256.  R0 holds elements 0..127, R1
    holds 128..255; the shared-index ``mult.ve`` (fixed_idx 0..255) spans both
    halves.  ``element_width`` is 1 (narrow, raw INT8 byte) or 4 (wide-vector
    debug, one INT32 lane per element -- MULT.VE/EE index ra_idx by lane).
    """
    num_blocks = math.ceil(channels / FPB)
    total = num_blocks * SUPER_BLOCK_BYTES * element_width
    packed = bytearray(total)

    def put(elem_idx: int, value: int) -> None:
        if element_width == 1:
            packed[elem_idx] = value & 0xFF
        else:
            struct.pack_into("<i", packed, elem_idx * 4, _as_signed_byte(value))

    for sb in range(num_blocks):
        sb_base = sb * SUPER_BLOCK_BYTES
        for s in range(FPB):
            ch = sb * FPB + s
            if ch >= channels:
                break
            slot = sb_base + s * CH_SLOT_BYTES
            for t in range(9):
                put(slot + t, kernel_raw[ch * 9 + t])
    return bytes(packed)


class DepthwiseConvUniversalApp(IpuApp):
    """Universal depthwise 3x3 convolution harness (no bias, no activation).

    Args:
        inst_path:    Path to assembled binary.
        input_path:   Path to input image binary (chunk-interleaved layout).
        kernel_path:  Path to kernel binary (channels * 9 bytes, INT8 raw).
        output_path:  Optional path to write output.
        dtype:        Data type string or :class:`DType`.
        rows:         Spatial height.
        cols:         Spatial width (16, 32, or 64).
        channels:     Number of channels (>= 1).
    """

    def __init__(
        self,
        *,
        dtype: str | DType = "INT8",
        rows: int,
        cols: int,
        channels: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.kernel_path = Path(self.kernel_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

        valid_cols = {16, 32, 64}
        if cols not in valid_cols:
            raise ValueError(f"cols must be in {valid_cols}, got {cols}")
        num_chunks = (rows * cols) // 128
        if num_chunks < 2:
            raise ValueError(
                f"Need at least 2 chunks (rows*cols >= 256), got {rows}*{cols}={rows*cols}"
            )
        if channels < 1:
            raise ValueError(f"channels ({channels}) must be >= 1")

        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.num_chunks = num_chunks
        # group_stride is a row-count (XMEM-space): one "chunk group" of
        # `channels` rows, one row per channel.
        self.group_stride = channels
        self.num_super_blocks = math.ceil(channels / FPB)
        self.total_kernel_rows = self.num_super_blocks * SUPER_BLOCK_ROWS
        # Guard band: the g0 section's kr=-1 prefetch computes
        # `lr8 - group_stride` at chunk 0, which must not go negative under
        # the 32-bit LR/CR range check (old 21-bit mask silently wrapped it;
        # it's masked-away don't-care data at the top border either way).
        # Fix: place the real input data one group_stride further into the
        # address space, and seed lr8 to group_stride so chunk 0 lands at
        # row 0 exactly.
        self.input_data_row = INPUT_BASE_ROW + self.group_stride
        self.input_data_addr = self.input_data_row * CHUNK_BYTES
        self._element_width = 1

    def setup(self, state: "IpuState") -> None:
        # Master ISA: dtype is a state attribute, not a CR register.
        state.dtype = self.dtype
        self._element_width = 4 if state.wide_vector_debug else 1
        row_bytes = CHUNK_BYTES * self._element_width

        input_data = self.input_path.read_bytes()
        state.xmem.write_address(self.input_data_row * row_bytes, input_data)

        kernel_raw = self.kernel_path.read_bytes()
        expected = self.channels * 9
        if len(kernel_raw) != expected:
            raise ValueError(
                f"kernel_path file has {len(kernel_raw)} bytes, "
                f"expected {expected} (channels * 9)"
            )
        kernel_packed = _pack_depthwise_kernel(
            kernel_raw, self.channels, self._element_width,
        )
        state.xmem.write_address(KERNEL_BASE_ROW * row_bytes, kernel_packed)

        # Border mask: a SINGLE blob carrying all 3 slots (0=none, 3=top-row
        # zero, 6=bottom-row zero), loaded once at init.  The g0 section selects
        # slot 3, the gN section selects slot 6 — no mid-program R_MASK reload.
        # Left/right edge columns are applied at runtime by mask_shift (CR15
        # partition below).  No zero region.
        state.xmem.write_address(MASK_BASE_ROW * row_bytes, build_border_mask_blob(self.cols))

        # CR15 dstructure: partition so each partition group is exactly one
        # packed spatial row (group size == cols).  The asm's mask_shift then
        # injects the left/right edge-column zero at each packed-row boundary.
        cols_to_partition = {
            64: Partition.P2,   # 2 groups of 64 lanes
            32: Partition.P4,   # 4 groups of 32 lanes
            16: Partition.P8,   # 8 groups of 16 lanes
        }
        if self.cols not in cols_to_partition:
            raise ValueError(
                f"depthwise_conv_universal mask-shift scheme requires "
                f"cols in {sorted(cols_to_partition)} (one packed row per partition "
                f"group); got cols={self.cols}"
            )
        state.set_cr_dstructure(
            valid_elements=128,
            partition=cols_to_partition[self.cols],
        )

        # CR map (master ISA: CR0 = read-only 0, CR1 = read-only 1, CR15 = dstructure).
        # Relocate the input/kernel bases off CR0/CR1 (mirroring
        # conv_universal_bn_activation), keeping CR0 free as the read-only zero
        # constant used by "SET lr<n>, cr0":
        #   CR10 = INPUT_BASE_ROW (cyclic-load base; the asm's own running
        #   pointer lr8/lr2 already carries the guard-band group_stride, which
        #   is why the DATA itself is written at input_data_row -- cr10 stays
        #   at the un-shifted base or the shift would be applied twice).
        #   CR5 = KERNEL_BASE (row), CR3 = mask blob row (single blob, slots 0/3/6).
        state.regfile.set_cr(10, INPUT_BASE_ROW)
        state.regfile.set_cr(5, KERNEL_BASE_ROW)
        # cr2 is pre-biased by -1 ROW for the deferred store (asm advances lr7
        # BEFORE the XMEM store at tap 2; store writes to lr7_advanced + cr2 =
        # lr7_old + OUTPUT_BASE_ROW).
        state.regfile.set_cr(2, (OUTPUT_BASE_ROW - 1) & 0xFFFFFFFF)
        state.regfile.set_cr(3, MASK_BASE_ROW)
        # cr9 = 384: r_cyclic slot-pointer step (+384 mod 512 ELEMENTS) for the
        # running write pointer lr5 -- element-space, unchanged by row addressing.
        state.regfile.set_cr(9, 384)

        # Parameter CR registers
        state.regfile.set_cr(4, self.cols)
        # cr6 = group_stride in ROWS (XMEM-space: feeds LDR_CYCLIC_MULT_REG's
        # offset/base sum via lr2/lr14, and the chunk/loop-limit comparisons).
        state.regfile.set_cr(6, self.group_stride)
        state.regfile.set_cr(7, FPB)               # channel group inner-loop size, in rows
        state.regfile.set_cr(8, self.total_kernel_rows)
        # cr11: chunk-loop limit, biased by the same guard-band group_stride
        # added to cr10/input_data_row above.
        state.regfile.set_cr(
            11, (self.num_chunks - 1) * self.group_stride + self.group_stride,
        )
        # cr12 = 128: r_cyclic slot size (ELEMENT-space; index step for lr5).
        # This is NOT the XMEM chunk stride anymore -- that role is CR1
        # (read-only 1: one XMEM chunk == one row).
        state.regfile.set_cr(12, 128)
        state.regfile.set_cr(13, 256)  # r_cyclic half-slot step, element-space
        state.regfile.set_cr(14, (256 - 2 * self.cols - 2) & 0xFFFFFFFF)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            total_outputs = self.num_chunks * self.channels
            dump_outputs(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_CHUNK_BYTES, total_outputs,
            )
