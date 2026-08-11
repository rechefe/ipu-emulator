"""attn@V (AGG kernel) harness — query-major scores, channel-major output.

Computes, per attention head h in [0,4):
    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 256),  t in [0, 36)

Inputs (channel-major activation tensors; .asm operands are ROW numbers,
issue #179):
  P query-major  : P[i, s] at PBASE + h*512 + i*2 + s//128 rows (4 heads, head-major)
  V channel-major: V[s, t] at VBASE + (h*36 + t)*2 + s//128 rows

Output (FP32 R_ACC, one row per group — same convention as the transformer
matmuls):
  O[i, t] at OBASE + (h*36 + t)*2 + g rows,  i = g*128 + local
  i.e. channel (h*36 + t) occupies 2 rows (2 groups of 128 FP32 lanes).

Usage::

    from ipu_apps.attn_v_256x36 import AttnV256x36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_TOK   = 256       # queries == keys
D       = 36        # head_dim
N_HEAD  = 4
N_CHAN  = N_HEAD * D  # 144 value channels total

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
PV_STRIDE_ROWS     = N_TOK // LANES          # 2: rows per P query / V channel
CHUNK_OFF_ROWS     = 1                       # second key chunk is the next row
P_GROUP1_OFF_ROWS  = (LANES * N_TOK) // LANES  # 256: P group-1 offset
P_HEAD_STRIDE_ROWS = N_TOK * PV_STRIDE_ROWS  # 512: rows per head in P
O_GROUP_ROWS       = 1                       # one r_acc store = one row (wide)
O_CHAN_ROWS        = 2 * O_GROUP_ROWS        # 2 groups per output channel

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


class AttnV256x36App(IpuApp):
    """attn@V AGG kernel harness (4 heads, N=256, head_dim=36)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are stored verbatim (already in the kernel's byte layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # CR1 (==1) is read-only hardwired; cr0 (==0) is hardwired zero.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(5, PV_STRIDE_ROWS)     # P query / V channel stride (rows)
        state.regfile.set_cr(6, CHUNK_OFF_ROWS)     # chunk offset (rows)
        state.regfile.set_cr(7, P_GROUP1_OFF_ROWS)  # P group-1 offset (rows)
        state.regfile.set_cr(8, P_HEAD_STRIDE_ROWS) # P head stride (rows)
        state.regfile.set_cr(9, 127)            # inner-loop bound
        state.regfile.set_cr(10, D)             # t count (36)
        state.regfile.set_cr(11, N_HEAD)        # head count (4)
        state.regfile.set_cr(12, O_GROUP_ROWS)      # O group stride (rows, FP32)
        state.regfile.set_cr(13, O_CHAN_ROWS)       # O channel stride (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # 144 channels, each 1024 B (two 512-B FP32 group rows).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )
