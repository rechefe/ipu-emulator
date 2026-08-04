"""Residual add 64×192 harness (Layer 4).

Computes C[r] = A[r] + B[r]  for r = 0..191,
where A and B are [64 tokens, 192 channels] in channel-major layout: one
channel per XMEM row, 64 valid tokens in the row's first 64 lanes. Rows are
never shared between channels, so each of the 192 channels owns a whole row
and the unused lanes 64..127 are cropped by the consumer.

Layer 4 parameters: d=192, N=64 tokens/stream, P=4, h=4, head_dim=48, L=4.

Usage::

    from ipu_apps.residual_add_64x192 import ResidualAdd64x192App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

N_CH   = 192                 # channels -> one XMEM row each
N_TOK  = 64                  # valid tokens per row (lanes 0..63)
N_ROWS = N_CH                # 192 vector rows, one per channel

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

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
OUTPUT_STRIDE_ROWS = 1
ROW_STRIDE_ROWS    = 1                       # one A/B vector row per XMEM row

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


class ResidualAdd64x192App(IpuApp):
    """64-token × 192-channel residual add application harness."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(self.input_a_path)
        self.input_b_path = Path(self.input_b_path)

    def setup(self, state: "IpuState") -> None:
        raw_a = Path(self.input_a_path).read_bytes()
        raw_b = Path(self.input_b_path).read_bytes()
        state.xmem.write_address(A_BASE, bytearray(raw_a))
        state.xmem.write_address(B_BASE, bytearray(raw_b))
        state.xmem.write_address(ONES_BASE, _ones_row())

        # CR1 (=1) is a read-only hardwired constant on the new architecture --
        # writes are silently dropped. B_BASE is therefore on CR9 (free).
        # cr0=A_BASE_ROW is 0 (harmless no-op, matches the hardwired 0).
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
        # pass-through multiplier that turns the multiplier into an adder.
        state.regfile.set_cr(10, 1)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_ROWS,
            )
