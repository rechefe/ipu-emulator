"""attn@V (broadcast kernel, Layer 5) harness — key-major scores, channel-major output.

Computes, per attention head h in [0,4):

    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 16),  t in [0, 60)

Layer-5 shape: d=240, N_TOK=16 tokens per stream, h=4 heads, head_dim=60.
This is the L5 port of ``attn_v_bcast_36``; the MAPPING is unchanged -- lanes
are QUERIES, V[s,t] is the broadcast scalar taken from R0, P key-major streams
through R_CYCLIC, and ACC.ADD accumulates over keys (no AGG). Only the counts
move:

  * ``D`` 36 -> 60 head channels per head (an outer loop count). head_dim is a
    loop count, not a lane count, so 60 needs no padding.
  * ``N_TOK`` 256 -> 16.  At L3 the 256 queries spanned TWO 128-lane groups
    (the g-loop) and a channel's 256 key values needed BOTH R0 and R1.  At L5
    all 16 queries fit in ONE group and a channel's 16 key values fit entirely
    in R0, so the g-loop and the R1 load both disappear.

This is the second half of the KEY-MAJOR chain (``attn_scores_km_16x60`` ->
``attn_v_bcast_60``).  The query-major sibling ``attn_v_16x60`` computes the
same mathematical product with a DIFFERENT datapath (MULT.RC.VV + AGG.SUM) and
is bit-different by design; the two chains share V's and O's layouts so their
outputs are directly comparable, but never their goldens.

ONE CHANNEL PER ROW: every P key row, every V channel row and every O channel
row is a WHOLE 512-B XMEM row holding 16 live FP32 lanes; rows are never
shared.  The consumer crops to the live lanes.

Inputs (wide FP32):
  P key-major    : P[i, s] at PBASE + h*P_HEAD_STRIDE_ROWS + s*PV_STRIDE_ROWS
                   (key s's 16 query scores = one row, query i in lane i)
  V channel-major: V[s, t] at VBASE + (h*60 + t)*PV_STRIDE_ROWS
                   (value channel's 16 keys = one row) -- IDENTICAL to
                   ``attn_v_16x60``'s V layout.

Output (FP32 R_ACC, one 512-B row per value channel -- IDENTICAL to
``attn_v_16x60``'s O layout):
  O[i, t] at OBASE + (h*60 + t)*O_CHAN_ROWS rows, lane i

Usage::

    from ipu_apps.attn_v_bcast_60 import AttnVBcast60App
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
# sub-row stride).  These match ``attn_v_16x60`` exactly for V and O.
PV_STRIDE_ROWS     = max(1, N_TOK // LANES)  # 1: rows per P key / V channel
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


class AttnVBcast60App(IpuApp):
    """attn@V broadcast kernel harness (4 heads, N=16, head_dim=60)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are stored verbatim (already in the kernel's row layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # Only N_TOK of the 128 lanes carry real queries. valid_elements gates
        # the ACTIVATE.QUANTIZE window, so a stored row holds exactly N_TOK
        # results and the padding lanes stay zero.
        state.set_cr_dstructure(valid_elements=N_TOK)

        # CR0 (==0) and CR1 (==1) are hardwired; the rest are writable.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(6, -1)                 # key-index startup
        state.regfile.set_cr(7, P_HEAD_STRIDE_ROWS) # P head stride (rows)
        # Loop bounds are count-1: the counter ADD and the BLT share one bundle,
        # so BLT reads the pre-ADD snapshot (branch taken while snapshot < bound).
        state.regfile.set_cr(8, N_TOK - 2)          # 14: key-loop bound (width 16, peeled+startup)
        state.regfile.set_cr(9, D - 1)              # 59: t-loop bound (60 channels)
        state.regfile.set_cr(10, N_HEAD - 1)        # 3: head-loop bound (4 heads)
        # LRs
        state.regfile.set_lr(0, 0)                  # r_cyclic index / mask_shift
        state.regfile.set_lr(1, PV_STRIDE_ROWS)     # P key stride / V channel stride (rows)
        state.regfile.set_lr(2, O_CHAN_ROWS)        # output-row stride (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N_CHAN channels, each ONE whole 512-B FP32 row (16 live lanes).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )
