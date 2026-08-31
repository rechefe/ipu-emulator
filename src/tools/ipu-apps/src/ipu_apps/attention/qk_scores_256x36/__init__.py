"""QKᵀ scores harness (Agent C), one attention head.

Computes the query-major score matrix for a single attention head::

    S[i, s] = sum_{c=0..35} Q[i, c] * K[s, c]      for i, s in [0, 256)

Inputs Q, K are logically channel-major (head_dim D=36, N=256 tokens). K is
loaded channel-major verbatim; Q is staged query-major (a gather of its strided
channels) so one query's 36 head-channels load into r0 with a single
``LDR_MULT_REG`` — the matmul broadcast template (scalar = Q[i,c] from r0,
vector = K's channel-c column in r_cyclic, ``MULT.RC.VE``).

The score row is stored RAW (full-precision R_ACC, 512 B per 128-key group,
query-major) so softmax (Agent A) reads unquantized scores. No AGG.

Wide-vector FP32 only: elements are 4-byte FP32 and an XMEM row is 512 B.

Usage::

    from ipu_apps.attention.qk_scores_256x36 import QkScores256x36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, ShapeBundle, no, yes
from ipu_apps.attention._spec_support import positive_dims, scores_query

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N    = 256          # tokens (queries = keys)
D    = 36           # head_dim (contraction width)
N_TG = 2            # key groups of 128 keys each
N_TPG = 128         # keys per group

# ---------------------------------------------------------------------------
# Wide-vector FP32 only. Elements are 4 bytes and an XMEM row is LANES * 4 =
# 512 B, unconditionally -- there is no narrow path. INT8 is not a mode this
# kernel is written against; it belongs at the XMEM write boundary.
#
# XMEM .asm operands are ROW numbers (issue #179). Region bases are DERIVED
# from row counts, not hardcoded bytes.
# ---------------------------------------------------------------------------
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

K_STRIDE_ROWS    = N // LANES                # 2: rows per K channel column
QROW_STRIDE_ROWS = 1                         # one staged query row per query
ACC_STORE_ROWS   = 1                         # one r_acc store = one row (wide)

K_ROWS    = D * K_STRIDE_ROWS
QROW_ROWS = N * QROW_STRIDE_ROWS
S_ROWS    = N * N_TG * ACC_STORE_ROWS

K_BASE_ROW    = 0
QROW_BASE_ROW = K_BASE_ROW + K_ROWS
S_BASE_ROW    = QROW_BASE_ROW + QROW_ROWS

K_BASE      = K_BASE_ROW * ROW_BYTES
QROW_BASE   = QROW_BASE_ROW * ROW_BYTES
S_BASE      = S_BASE_ROW * ROW_BYTES

QROW_STRIDE      = QROW_STRIDE_ROWS * ROW_BYTES
OUTPUT_ROW_BYTES = 512


class QkScores256x36App(IpuApp):
    """One-head QKᵀ → query-major scores application harness (wide FP32)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.query_path = Path(self.query_path)
        self.key_path = Path(self.key_path)

    # -- staging -------------------------------------------------------------

    def _stage_inputs(self, state: "IpuState") -> None:
        """Write K channel-major and Q query-major into XMEM.

        Input files are stored channel-major: element [token t, channel c] at
        (c*N + t)*ELEM_BYTES.
        """
        q_raw = self.query_path.read_bytes()
        k_raw = self.key_path.read_bytes()

        # K: channel-major verbatim. Column c (256 keys) is contiguous already;
        #    write at K_BASE + c*(N*ELEM_BYTES). The kernel loads two 128-key chunks.
        for c in range(D):
            col = k_raw[(c * N) * ELEM_BYTES : (c * N + N) * ELEM_BYTES]
            state.xmem.write_address(K_BASE + c * K_STRIDE_ROWS * ROW_BYTES, bytearray(col))

        # Q: gather the strided channels into contiguous query-major rows.
        #    QROW[i] = Q[i, 0..35] at QROW_BASE + i*QROW_STRIDE (rest zero-pad).
        for i in range(N):
            row = bytearray(QROW_STRIDE)
            for c in range(D):
                src = (c * N + i) * ELEM_BYTES
                row[c * ELEM_BYTES : (c + 1) * ELEM_BYTES] = q_raw[src : src + ELEM_BYTES]
            state.xmem.write_address(QROW_BASE + i * QROW_STRIDE, row)

    def setup(self, state: "IpuState") -> None:
        self._stage_inputs(state)

        # Startup skews, in rows. These are lane counts, so they carry no
        # element-width factor.
        g0_start_rows = -K_STRIDE_ROWS                       # g=0: first live = row 0
        g1_start_rows = -K_STRIDE_ROWS + N_TPG // LANES      # g=1: first live = +1 row

        # CR1 (≡1) is read-only hardwired; QROW base lives on CR9.
        state.regfile.set_cr(0, K_BASE_ROW)             # data base
        state.regfile.set_cr(9, QROW_BASE_ROW)          # staged query rows
        state.regfile.set_cr(3, S_BASE_ROW)             # group 0 output base
        state.regfile.set_cr(4, S_BASE_ROW + ACC_STORE_ROWS)   # group 1 output base
        state.regfile.set_cr(5, g0_start_rows)           # g=0 K-data startup (rows)
        state.regfile.set_cr(6, g1_start_rows)           # g=1 K-data startup (rows)
        state.regfile.set_cr(7, -1)                      # channel fixed_idx startup
        state.regfile.set_cr(8, D - 2)                   # contraction bound (34)

        state.regfile.set_lr(0, 0)                       # r_cyclic write-index / mask_shift
        state.regfile.set_lr(2, K_STRIDE_ROWS)           # K data stride per channel (rows)
        state.regfile.set_lr(3, N_TG * ACC_STORE_ROWS)   # output stride per query (rows)
        state.regfile.set_lr(6, D - 2)                   # contraction BLT bound
        state.regfile.set_lr(7, 0)                       # output query row offset
        state.regfile.set_lr(8, 0)                       # Q-row row offset
        state.regfile.set_lr(9, 0)                       # query counter
        state.regfile.set_lr(10, N)                      # query-loop limit
        state.regfile.set_lr(12, QROW_STRIDE_ROWS)       # Q-row stride per query (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N queries × N_TG groups × 512 B, in query-major group order:
            #   row (i, g) at S_BASE_ROW + i*N_TG + g, in rows.
            dump_xmem_to_binary(
                state, self.output_path,
                S_BASE, OUTPUT_ROW_BYTES, N * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's exact-match domain
# (n_tok=256, d=36). No k_base_row-style invariant here -- K_BASE_ROW is 0 by
# construction but the constructor takes no caller-supplied base-row kwargs to
# guard against.


def _supports(**params):
    q = scores_query(n_tok=params["n_tok"], d=params["d"])
    bad = positive_dims(n_tok=q.n_tok, d=q.d)
    if bad:
        return no(bad)
    if q.n_tok != N:
        return no(f"handles exactly n_tok={N}; got {q.n_tok}")
    if q.d != D:
        return no(f"handles exactly d={D}; got {q.d}")
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"n_tok == {N} and d == {D} exactly: the one-head QKT query-major "
        f"scores kernel, 256 tokens spanning {N_TG} 128-lane key groups."
    )


SPEC = KernelSpec(
    name="qk_scores_256x36",
    op="qk_scores",
    variant="256x36",
    app_class=QkScores256x36App,
    asm="qk_scores_256x36.asm",
    requires=("n_tok", "d"),
    tags=("fp32-wide", "query-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        query=(params["n_tok"], params["d"]), key=(params["n_tok"], params["d"])
    ).with_shapes(derived={"output": (params["n_tok"], params["n_tok"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
