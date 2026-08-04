"""Residual add 16x240 harness (L5).

Computes C[ch] = A[ch] + B[ch] for ch = 0..239, where A and B are
[240 channels, 16 tokens] in channel-major layout.

ONE CHANNEL PER ROW. N_TOK=16 means a channel's tokens are 16 FP32 values
(64 bytes) in a 512-byte XMEM row -- rows are NEVER shared between channels.
The remaining 112 lanes are padding on input and stale-but-zero on output;
``teardown`` crops each output row to its 16-token prefix. Packing several
channels into one row at a 64-byte stride would be a bug (it was one, once).

Usage::

    from ipu_apps.residual_add_16x240 import ResidualAdd16x240App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_CH   = 240                 # channels
N_TOK  = 16                  # tokens per channel (one token group)
N_ROWS = N_CH                # one XMEM row per channel

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (issue #179). Region bases are DERIVED
# from row counts, not hardcoded bytes: a byte map sized for 1-byte elements
# overflows at 4 bytes/element and regions silently overwrite each other.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

# One r_acc store is 512 B = exactly one row in wide mode.
OUTPUT_ROW_BYTES   = ROW_BYTES
VALID_ROW_BYTES    = N_TOK * ELEM_BYTES      # 64 B of meaningful output per row
OUTPUT_STRIDE_ROWS = 1
ROW_STRIDE_ROWS    = 1                       # one A/B channel row per XMEM row

ONES_ROWS = 1
A_BASE_ROW      = 0
B_BASE_ROW      = A_BASE_ROW + N_ROWS
ONES_BASE_ROW   = B_BASE_ROW + N_ROWS
OUTPUT_BASE_ROW = ONES_BASE_ROW + ONES_ROWS

A_BASE      = A_BASE_ROW * ROW_BYTES
B_BASE      = B_BASE_ROW * ROW_BYTES
ONES_BASE   = ONES_BASE_ROW * ROW_BYTES
OUTPUT_BASE = OUTPUT_BASE_ROW * ROW_BYTES


def _ones_row() -> bytearray:
    """One XMEM row of FP32 1.0 -- the pass-through multiplier vector."""
    return bytearray(np.ones(LANES, dtype=np.float32).tobytes())


class ResidualAdd16x240App(IpuApp):
    """16-token x 240-channel residual add, wide-vector FP32."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(self.input_a_path)
        self.input_b_path = Path(self.input_b_path)

    def setup(self, state: "IpuState") -> None:
        # Inputs are full 512-byte rows (one channel each), so the raw bytes
        # land directly -- there is no element-index arithmetic to get wrong.
        raw_a = Path(self.input_a_path).read_bytes()
        raw_b = Path(self.input_b_path).read_bytes()
        assert len(raw_a) == N_ROWS * ROW_BYTES, (
            f"A is {len(raw_a)} bytes, expected {N_ROWS * ROW_BYTES} "
            "(one full 512-byte row per channel)"
        )
        assert len(raw_b) == N_ROWS * ROW_BYTES, (
            f"B is {len(raw_b)} bytes, expected {N_ROWS * ROW_BYTES}"
        )
        state.xmem.write_address(A_BASE, bytearray(raw_a))
        state.xmem.write_address(B_BASE, bytearray(raw_b))
        state.xmem.write_address(ONES_BASE, _ones_row())

        # CR0 (=0) and CR1 (=1) are read-only hardwired constants; writes are
        # silently dropped. A_BASE_ROW is 0 so CR0 is a harmless no-op, and
        # B_BASE lives on CR9.
        state.regfile.set_cr(0, A_BASE_ROW)
        state.regfile.set_cr(9, B_BASE_ROW)
        state.regfile.set_cr(2, ONES_BASE_ROW)
        state.regfile.set_cr(3, OUTPUT_BASE_ROW)
        state.regfile.set_cr(4, 0)
        state.regfile.set_cr(5, -ROW_STRIDE_ROWS)          # A/B ptr startup: -1 row
        state.regfile.set_cr(6, N_ROWS)
        state.regfile.set_cr(7, ROW_STRIDE_ROWS)           # A/B row stride (rows)
        state.regfile.set_cr(8, OUTPUT_STRIDE_ROWS)        # output row stride (rows)
        # cr10 = 1: in wide FP32 a CR scalar is its low byte read as a signed
        # int and converted to float, so 1 gives exactly 1.0 -- the MULT.RC.VE
        # pass-through multiplier.
        state.regfile.set_cr(10, 1)

    def teardown(self, state: "IpuState") -> None:
        """Dump the output, cropping each row to its N_TOK valid tokens.

        Each channel owns a whole row; only the first N_TOK lanes are meaningful.
        """
        if self.output_path is None:
            return
        raw_path = Path(self.output_path).with_suffix(".rows.bin")
        dump_xmem_to_binary(
            state, raw_path,
            OUTPUT_BASE, OUTPUT_ROW_BYTES, N_ROWS,
        )
        rows = np.frombuffer(raw_path.read_bytes(), dtype=np.float32).reshape(
            N_ROWS, LANES
        )
        Path(self.output_path).write_bytes(
            np.ascontiguousarray(rows[:, :N_TOK]).tobytes()
        )
