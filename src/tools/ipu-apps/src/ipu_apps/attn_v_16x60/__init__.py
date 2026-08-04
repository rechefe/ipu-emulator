"""attn@V (AGG kernel, Layer 5) harness — query-major scores, channel-major output.

Computes, per attention head h in [0,4):
    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 16),  t in [0, 60)

Layer-5 shape: d=240, N_TOK=16 tokens per stream, h=4 heads, head_dim=60.
This is the L5 port of ``attn_v_256x36``; the mapping (query-major P, lanes =
keys, ``MULT.RC.VV`` + ``AGG.SUM``) is unchanged. Only the counts move:

  * ``D`` 36 -> 60 head channels per head (an outer loop count).
  * ``N_TOK`` 256 -> 16.  At L3 the 256 keys spanned TWO 128-lane chunks
    (``AGG.SUM.FIRST`` then ``AGG.SUM``) and the 256 queries spanned two
    groups.  At L5 all 16 keys fit in ONE chunk and all 16 queries fit in ONE
    R_ACC store, so three of the four L3 inner loops disappear: a single
    ``AGG.SUM.FIRST`` loop of 16 bundles produces a whole output row.

ONE CHANNEL PER ROW: every P query row and every V channel row is a WHOLE
512-B XMEM row holding 16 live FP32 elements; rows are never shared. The
output row for channel (h*60 + t) is likewise one whole row with 16 live lanes,
cropped by the consumer.

Inputs (wide FP32, channel-major activation tensors):
  P query-major  : P[i, s] at PBASE + h*P_HEAD_STRIDE + i*PV_STRIDE (rows)
  V channel-major: V[s, t] at VBASE + (h*60 + t)*PV_STRIDE (rows)

Output (FP32 R_ACC, one 512-B row per value channel):
  O[i, t] at OBASE + (h*60 + t)*O_CHAN_ROWS rows, lane i

Usage::

    from ipu_apps.attn_v_16x60 import AttnV16x60App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_TOK   = 16        # queries == keys
D       = 60        # head_dim
N_HEAD  = 4
N_CHAN  = N_HEAD * D  # 240 value channels total

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

# Row counts below are LANE counts -- element-width independent by construction.
# One channel per row: 16 tokens live in a WHOLE row (ceil-div, never a
# sub-row stride).
PV_STRIDE_ROWS     = max(1, N_TOK // LANES)  # 1: rows per P query / V channel
P_HEAD_STRIDE_ROWS = N_TOK * PV_STRIDE_ROWS  # 16: rows per head in P
O_CHAN_ROWS        = 1                       # one r_acc store = one whole row

P_ROWS = N_HEAD * P_HEAD_STRIDE_ROWS
V_ROWS = N_CHAN * PV_STRIDE_ROWS
O_ROWS = N_CHAN * O_CHAN_ROWS

PBASE_ROW = 0
VBASE_ROW = PBASE_ROW + P_ROWS
OBASE_ROW = VBASE_ROW + V_ROWS

PBASE = PBASE_ROW * ROW_BYTES
VBASE = VBASE_ROW * ROW_BYTES
OBASE = OBASE_ROW * ROW_BYTES

P_HEAD_STRIDE = P_HEAD_STRIDE_ROWS * ROW_BYTES
O_CHAN_BYTES  = O_CHAN_ROWS * ROW_BYTES


class AttnV16x60App(IpuApp):
    """attn@V AGG kernel harness (4 heads, N=16, head_dim=60)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are stored verbatim (already in the kernel's row layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # Only N_TOK of the 128 lanes carry real keys. AGG.SUM reduces exactly
        # valid_elements lanes, and ACTIVATE.QUANTIZE writes exactly that many
        # output lanes -- one setting covers both.
        state.set_cr_dstructure(valid_elements=N_TOK)

        # CR1 (==1) is read-only hardwired; cr0 (==0) is hardwired zero.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(5, PV_STRIDE_ROWS)     # P query / V channel stride (rows)
        state.regfile.set_cr(8, P_HEAD_STRIDE_ROWS) # P head stride (rows)
        # Inner bound is count-1: the INC and the BLT share one bundle, so BLT
        # reads the pre-INC snapshot.
        state.regfile.set_cr(9, N_TOK - 1)      # 15: inner-loop bound -> 16 AGG bundles
        state.regfile.set_cr(10, D)             # t count (60)
        state.regfile.set_cr(11, N_HEAD)        # head count (4)
        state.regfile.set_cr(13, O_CHAN_ROWS)       # O channel stride (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N_CHAN channels, each ONE whole 512-B FP32 row (16 live lanes).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )
