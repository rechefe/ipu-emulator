"""QK^T scores harness (Layer 4), query-major, per (stream, head) block.

Computes the query-major score matrix for every (stream, head) pair of the
Layer-4 attention block::

    S[b, i, s] = sum_{c=0..47} Q[b, i, c] * K[b, s, c]     i, s in [0, 64)

where ``b`` enumerates the ``P * N_HEAD = 16`` (stream, head) blocks.  Layer 4
parameters: d = 192, N = 64 tokens per stream, P = 4 streams, h = 4 heads,
head_dim = 48.  head_dim = 48 needs no padding -- it is a loop count (the
contraction bound), not a lane count.

This is the L4 port of ``qk_scores_256x36``.  The mapping is carried over
unchanged: K is loaded channel-major (one channel column into R_CYCLIC), Q is
staged query-major so that one ``LDR_MULT_REG`` puts a query's 48 head-channels
into R0, and ``MULT.RC.VE`` broadcasts the scalar Q[i,c] over the key column.
No AGG.

The one structural difference from L3 is that N = 64 < LANES = 128, so a whole
key axis fits in a single 128-lane row: there is exactly ONE key group
(``N_TG = 1``) instead of two, and the trailing 64 lanes of every K / S row are
unused padding.  Per the one-channel-per-row rule an XMEM row is still never
shared -- each (block, query) score row owns a whole row and the teardown crops
the valid ``N * ELEM_BYTES`` prefix out of it.

It pairs with ``attn_v_64x48`` (query-major scores -> attn@V).  It must NOT be
paired with ``attn_v_bcast_48``, which consumes key-major scores.

Usage::

    from ipu_apps.qk_scores_64x48 import QkScores64x48App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N       = 64            # tokens per stream (queries == keys)
D       = 48            # head_dim (contraction width)
P       = 4             # streams / partitions
N_HEAD  = 4             # attention heads
N_BLOCK = P * N_HEAD    # 16 independent (stream, head) score problems
N_TG    = 1             # key groups: 64 keys fit in one 128-lane row
N_TPG   = N            # keys per group

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (issue #179). Region bases are DERIVED
# from row counts, not hardcoded bytes: a byte map sized for 1-byte elements
# overflows 4x at FP32 and silently corrupts results.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

# Row counts are LANE counts -- element-width independent by construction.
# N = 64 <= LANES, so one K channel column is one whole row (tail unused).
K_STRIDE_ROWS    = 1                         # rows per K channel column
QROW_STRIDE_ROWS = 1                         # one staged query row per query
ACC_STORE_ROWS   = 1                         # one r_acc store = one row (wide)

K_BLOCK_ROWS = D * K_STRIDE_ROWS             # 48 rows per (stream, head) block
Q_BLOCK_ROWS = N * QROW_STRIDE_ROWS          # 64 rows per block
S_BLOCK_ROWS = N * N_TG * ACC_STORE_ROWS     # 64 rows per block

K_ROWS    = N_BLOCK * K_BLOCK_ROWS           # 768
QROW_ROWS = N_BLOCK * Q_BLOCK_ROWS           # 1024
S_ROWS    = N_BLOCK * S_BLOCK_ROWS           # 1024

K_BASE_ROW    = 0
QROW_BASE_ROW = K_BASE_ROW + K_ROWS
S_BASE_ROW    = QROW_BASE_ROW + QROW_ROWS

K_BASE    = K_BASE_ROW * ROW_BYTES
QROW_BASE = QROW_BASE_ROW * ROW_BYTES
S_BASE    = S_BASE_ROW * ROW_BYTES

QROW_STRIDE = QROW_STRIDE_ROWS * ROW_BYTES
K_STRIDE    = K_STRIDE_ROWS * ROW_BYTES

# One store = one whole row; only the first N lanes are valid scores.
OUTPUT_ROW_BYTES = N * ELEM_BYTES            # 256 valid bytes per stored row


class QkScores64x48App(IpuApp):
    """Layer-4 QK^T -> query-major scores, 16 (stream, head) blocks (wide FP32)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.query_path = Path(self.query_path)
        self.key_path = Path(self.key_path)

    # -- staging -------------------------------------------------------------

    def _stage_inputs(self, state: "IpuState") -> None:
        """Write K channel-major and Q query-major into XMEM.

        Input files are channel-major per block: element [block b, token t,
        channel c] at ((b*D + c)*N + t) * ELEM_BYTES.
        """
        q = np.frombuffer(self.query_path.read_bytes(), dtype=np.float32)
        k = np.frombuffer(self.key_path.read_bytes(), dtype=np.float32)
        want = N_BLOCK * D * N
        if q.size != want or k.size != want:
            raise ValueError(
                f"expected {want} FP32 elements in each of Q/K; "
                f"got {q.size} and {k.size}"
            )
        q = q.reshape(N_BLOCK, D, N)          # [block, channel, token]
        k = k.reshape(N_BLOCK, D, N)

        # K: one channel column per row, 64 keys in the first 64 lanes.
        k_buf = np.zeros((N_BLOCK * D, LANES), dtype=np.float32)
        for b in range(N_BLOCK):
            k_buf[b * D:(b + 1) * D, :N] = k[b]
        state.xmem.write_address(K_BASE, bytearray(k_buf.tobytes()))

        # Q: gather the strided channels into contiguous query-major rows.
        #    QROW[b, i] = Q[b, i, 0..47] at the row start, rest zero.
        q_buf = np.zeros((N_BLOCK * N, LANES), dtype=np.float32)
        for b in range(N_BLOCK):
            q_buf[b * N:(b + 1) * N, :D] = q[b].T   # [token, channel]
        state.xmem.write_address(QROW_BASE, bytearray(q_buf.tobytes()))

    def setup(self, state: "IpuState") -> None:
        self._stage_inputs(state)

        # Only N of the 128 lanes carry real keys; the rest must be excluded
        # structurally rather than relying on the harness zero-filling them.
        state.set_cr_dstructure(valid_elements=N)

        # Startup skew, in rows: the load runs one bundle ahead of the MULT that
        # consumes it, so the pointer starts one channel stride low.
        k_start_rows = -K_STRIDE_ROWS        # first live K column = block base

        # CR0 (==0) and CR1 (==1) are hardwired read-only, so every base lives
        # on a writable CR -- K_BASE_ROW happening to be 0 must not be relied on.
        state.regfile.set_cr(2, K_BASE_ROW)             # channel-major K base
        state.regfile.set_cr(9, QROW_BASE_ROW)          # staged query rows
        state.regfile.set_cr(3, S_BASE_ROW)             # score output base
        state.regfile.set_cr(5, k_start_rows)           # K-data startup (rows)
        state.regfile.set_cr(7, -1)                     # channel fixed_idx startup
        state.regfile.set_cr(8, D - 2)                  # contraction bound (46)
        state.regfile.set_cr(10, N_BLOCK)               # block-loop limit (16)
        state.regfile.set_cr(11, K_BLOCK_ROWS)          # K rows per block (48)

        state.regfile.set_lr(0, 0)                      # r_cyclic write-index / mask_shift
        state.regfile.set_lr(2, K_STRIDE_ROWS)          # K stride per channel (rows)
        state.regfile.set_lr(3, N_TG * ACC_STORE_ROWS)  # output stride per query (rows)
        state.regfile.set_lr(6, D - 2)                  # contraction BLT bound
        state.regfile.set_lr(7, 0)                      # output row offset
        state.regfile.set_lr(8, 0)                      # Q-row row offset
        state.regfile.set_lr(9, 0)                      # query counter
        state.regfile.set_lr(10, N)                     # query-loop limit
        state.regfile.set_lr(12, QROW_STRIDE_ROWS)      # Q-row stride per query (rows)
        state.regfile.set_lr(13, 0)                     # K block base offset (rows)
        state.regfile.set_lr(14, 0)                     # block counter

    def teardown(self, state: "IpuState") -> None:
        """Crop each score row's valid N keys out of its whole 512 B row.

        Every store wrote a full row (one query per row -- rows are never
        shared), but only the leading ``N * ELEM_BYTES`` bytes hold scores. The
        output file is the densely packed crop, ``N_BLOCK * N`` rows of N FP32.
        """
        if self.output_path is None:
            return
        stride = ACC_STORE_ROWS * ROW_BYTES
        parts = [
            bytes(state.xmem.read_address(S_BASE + r * stride, OUTPUT_ROW_BYTES))
            for r in range(N_BLOCK * N * N_TG)
        ]
        Path(self.output_path).write_bytes(b"".join(parts))
