"""kQᵀ → key-major attention scores, one head (Layer 5).

Computes, for a single attention head h (head_dim D=60, N=16 tokens)::

    S[i, s] = sum_c Q[i, c] * K[s, c]        i, s in [0, 16), c in [0, 60)

and stores S **key-major**: key s's score column S[:, s] occupies one WHOLE
XMEM row (16 live FP32 lanes, one query per lane), so the downstream softmax
chain reads a key column contiguously.

Layer-5 shape: d=240, N_TOK=16 tokens per stream, h=4 heads, head_dim=60.
This is the L5 port of ``attn_scores_km_256x36``; the mapping is unchanged and
only the loop counts move:

  * ``D`` 36 -> 60.  head_dim is the contraction LOOP COUNT here
    (``chan_bound = D-2``), not a lane count, so 60 needs no padding.
  * ``N_TOK`` 256 -> 16.  256 queries spanned two 128-lane query groups; 16
    queries fit in ONE group, so the g=1 half of the L3 kernel is gone
    (``N_TG`` 2 -> 1) and each key produces exactly one stored row.

The 4-head-input / one-head-selected interface is preserved: the input files
hold all ``N_HEADS`` heads in the canonical channel-major layout, and the
``head`` constructor argument picks which one this invocation computes.

Activation layout (canonical, channel-major, multi-head)::

    Q/K element [token t, head-channel c of head h] lives at
        element index (h*60 + c) * N_TOK + t

The kernel needs K[s, 0:59] contiguous to load into R0 (the scalar operand),
so :func:`_load_k_keymajor` rearranges this head's K into a key-major XMEM
scratch (K[s, :] in row s).  Q stays channel-major: each inner-loop step loads
a channel column (16 queries, one channel) straight into R_CYCLIC.

ONE CHANNEL PER ROW: a Q channel column, a key-major K row, and an output key
column are each a WHOLE 512-B row with 16 live FP32 lanes. Rows are never
shared; the consumer crops to the live lanes.

Usage::

    from ipu_apps.attention.attn_scores_km_16x60 import AttnScoresKM16x60App
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

N_TOK    = 16           # queries = keys
D        = 60           # head_dim
N_TG     = 1            # query groups: 16 queries fit in a single 128-lane group
N_TPG    = N_TOK        # queries per group
N_HEADS  = 4            # channels in the canonical input file = N_HEADS * D

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

# One channel per row: 16 tokens live in a WHOLE row (ceil-div, never a
# sub-row stride).
Q_CHAN_ROWS   = max(1, N_TOK // LANES)       # 1: rows per Q channel column
K_STRIDE_ROWS = 1                            # one key-major K row per key
OUT_ROWS      = 1                            # one r_acc store = one row (wide)

Q_ROWS = D * Q_CHAN_ROWS
K_ROWS = N_TOK * K_STRIDE_ROWS
S_ROWS = N_TOK * N_TG * OUT_ROWS

QBASE_ROW    = 0
KBASE_KM_ROW = QBASE_ROW + Q_ROWS
SBASE_ROW    = KBASE_KM_ROW + K_ROWS

QBASE    = QBASE_ROW * ROW_BYTES
KBASE_KM = KBASE_KM_ROW * ROW_BYTES
SBASE    = SBASE_ROW * ROW_BYTES

K_STRIDE = K_STRIDE_ROWS * ROW_BYTES
OUTPUT_ROW_BYTES = ROW_BYTES                 # r_acc store payload


def _load_q_channel_major(state: "IpuState", q_path: str | Path, head: int) -> None:
    """Copy head `head`'s D channel columns of Q into XMEM at QBASE, one per row.

    Input file is canonical channel-major: Q[t, h*D+c] at element (h*D+c)*N_TOK + t.
    The kernel addresses this head as QBASE + c*Q_CHAN_ROWS rows. A channel column
    is only N_TOK elements -- a FRACTION of a row -- so each column is written to
    its OWN whole row rather than copied as one contiguous block (one channel per
    row; a packed 64-B stride would be a bug).
    """
    raw = Path(q_path).read_bytes()
    base = head * D * N_TOK * ELEM_BYTES
    span = D * N_TOK * ELEM_BYTES
    if len(raw) < base + span:
        raise ValueError(
            f"{q_path}: expected >= {base + span} B for head {head}, got {len(raw)}"
        )
    q = np.frombuffer(raw[base : base + span], dtype=np.float32).reshape(D, N_TOK)
    for c in range(D):
        row = np.zeros(LANES, dtype=np.float32)
        row[:N_TOK] = q[c, :]                 # channel c's N_TOK queries
        state.xmem.write_address(
            QBASE + c * Q_CHAN_ROWS * ROW_BYTES, bytearray(row.tobytes())
        )


def _load_k_keymajor(state: "IpuState", k_path: str | Path, head: int) -> None:
    """Rearrange head `head`'s K from channel-major into key-major XMEM scratch.

    Source K[s, c] at element (head*D + c)*N_TOK + s.  Destination K[s, :]
    contiguous at KBASE_KM + s*K_STRIDE (D channels, zero-padded to a full row),
    which is what lets one LDR pull a key's whole head-channel vector into R0.
    """
    raw = Path(k_path).read_bytes()
    head_base = head * D * N_TOK * ELEM_BYTES
    span = D * N_TOK * ELEM_BYTES
    if len(raw) < head_base + span:
        raise ValueError(
            f"{k_path}: expected >= {head_base + span} B for head {head}, got {len(raw)}"
        )
    k = np.frombuffer(
        raw[head_base : head_base + span], dtype=np.float32
    ).reshape(D, N_TOK)                       # [channel, key]
    for s in range(N_TOK):
        row = np.zeros(LANES, dtype=np.float32)
        row[:D] = k[:, s]                     # key s's D head-channels
        state.xmem.write_address(KBASE_KM + s * K_STRIDE, bytearray(row.tobytes()))


class AttnScoresKM16x60App(IpuApp):
    """kQᵀ → key-major scores, single head (D=60, N=16), 4-head input."""

    def __init__(self, *, head: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        # Q is the "input"; K is the "weights" file slot.
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        if not 0 <= head < N_HEADS:
            raise ValueError(f"head must be in [0, {N_HEADS}), got {head}")
        self.head = head

    def setup(self, state: "IpuState") -> None:
        _load_q_channel_major(state, self.input_path, self.head)
        _load_k_keymajor(state, self.weights_path, self.head)

        # Only N_TOK of the 128 lanes carry real queries. valid_elements gates
        # the ACTIVATE.QUANTIZE window, so a stored row holds exactly N_TOK scores.
        state.set_cr_dstructure(valid_elements=N_TOK)

        # CR1 (≡1) is read-only hardwired; cr0 (=QBASE_ROW=0) matches hardwired 0.
        state.regfile.set_cr(0, QBASE_ROW)
        state.regfile.set_cr(2, SBASE_ROW)
        state.regfile.set_cr(9, KBASE_KM_ROW)
        # Startup skews are negative; CRs are 32-bit unsigned, and the kernel
        # only ever ADDs them back to a non-negative pointer, so wraparound is
        # exact modulo 2**32.
        state.regfile.set_cr(5, -Q_CHAN_ROWS)   # channel-column startup (rows)
        state.regfile.set_cr(7, -1)          # fixed_idx c startup
        state.regfile.set_cr(8, D - 2)       # c-loop bound: first=0, width=D → D-2 = 58

        state.regfile.set_lr(0, 0)           # R_CYCLIC index 0
        state.regfile.set_lr(2, Q_CHAN_ROWS)    # channel stride in Q (rows)
        state.regfile.set_lr(3, OUT_ROWS)       # output store stride (rows)
        state.regfile.set_lr(6, D - 2)       # c-loop bound = 58
        state.regfile.set_lr(7, 0)           # output row pointer
        state.regfile.set_lr(8, -K_STRIDE_ROWS) # key row offset startup (-1 -> first live 0)
        state.regfile.set_lr(9, 0)           # key counter
        state.regfile.set_lr(10, N_TOK)      # key-loop limit
        state.regfile.set_lr(12, K_STRIDE_ROWS) # key stride into K scratch (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # N_TOK keys x N_TG groups, one WHOLE row each (16 live lanes).
            dump_xmem_to_binary(
                state, self.output_path,
                SBASE, OUTPUT_ROW_BYTES, N_TOK * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's exact-match shape domain
# (n_tok=16, d=60). `head` is deliberately left OUT of `requires`/`supports`:
# it selects which of the 4 heads already present in the input file to
# compute, not a shape the query routes on -- every head is equally supported
# by the same kernel, so a query that never mentions `head` still gets a
# correct verdict. `build` does not surface it either; callers that need a
# specific head keep passing it straight to the constructor (default 0),
# which is unchanged, and (like its `attn_scores_km_64x48` sibling) now
# range-checks `head` against N_HEADS at construction time.


def _supports(**params):
    q = scores_query(n_tok=params["n_tok"], d=params["d"])
    bad = positive_dims(n_tok=q.n_tok, d=q.d)
    if bad:
        return no(bad)
    if q.n_tok != N_TOK:
        return no(f"handles exactly n_tok={N_TOK}; got {q.n_tok}")
    if q.d != D:
        return no(f"handles exactly d={D}; got {q.d}")
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"n_tok == {N_TOK} and d == {D} exactly: the L5 key-major kQT scores "
        f"kernel, one selected head of {N_HEADS} out of a 4-head input file."
    )


SPEC = KernelSpec(
    name="attn_scores_km_16x60",
    op="attn_scores_km",
    variant="16x60",
    app_class=AttnScoresKM16x60App,
    asm="attn_scores_km_16x60.asm",
    requires=("n_tok", "d"),
    tags=("fp32-wide", "key-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        query=(params["n_tok"], params["d"]), key=(params["n_tok"], params["d"])
    ).with_shapes(derived={"output": (params["n_tok"], params["n_tok"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
