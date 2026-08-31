"""Agent D — kQᵀ → key-major attention scores (one head).

Computes, for a single attention head h (head_dim D=36, N=256 tokens)::

    S[i, s] = sum_c Q[i, c] * K[s, c]        i, s in [0, 256), c in [0, 36)

and stores S **key-major**: S[i, s] at SBASE + (s*256 + i)*4 (int32/fp32 words),
so that each key column S[:, s] is contiguous (feeds the downstream softmax chain).

Activation layout (canonical, channel-major, multi-head)::

    Q/K element [token t, head-channel c of head h] lives at
        BASE + (h*36 + c) * 256 + t

The kernel needs K[s, 0:35] contiguous to load into R0 (the scalar operand),
so :func:`_load_k_keymajor` rearranges this head's K into a key-major XMEM
scratch (K[s, :] at KBASE_KM + s*128).  Q stays channel-major: each inner-loop
step loads a channel column (128 queries, one channel) straight into R_CYCLIC.

Usage::

    from ipu_apps.attention.attn_scores_km_256x36 import AttnScoresKM256x36App
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

N_TOK    = 256          # queries = keys
D        = 36           # head_dim
N_TG     = 2            # query groups of 128
N_TPG    = 128
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

Q_CHAN_ROWS   = N_TOK // LANES               # 2: rows per Q channel column
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
OUTPUT_ROW_BYTES = 512                       # r_acc store payload


def _load_q_channel_major(state: "IpuState", q_path: str | Path, head: int) -> None:
    """Copy head `head`'s 36 channel columns of Q verbatim into XMEM at QBASE.

    Input file is canonical channel-major: Q[t, h*36+c] at element (h*36+c)*256 + t.
    The kernel addresses this head as QBASE + c*Q_CHAN_ROWS rows, so we slice the
    head's contiguous channel block (channels h*36 .. h*36+35) to the front.
    Each channel column is N_TOK elements = Q_CHAN_ROWS whole rows, so the block
    is already row-aligned and copies verbatim.
    """
    raw = Path(q_path).read_bytes()
    base = head * D * N_TOK * ELEM_BYTES
    span = D * N_TOK * ELEM_BYTES
    if len(raw) < base + span:
        raise ValueError(
            f"{q_path}: expected >= {base + span} B for head {head}, got {len(raw)}"
        )
    state.xmem.write_address(QBASE, bytearray(raw[base : base + span]))


def _load_k_keymajor(state: "IpuState", k_path: str | Path, head: int) -> None:
    """Rearrange head `head`'s K from channel-major into key-major XMEM scratch.

    Source K[s, c] at element (head*36 + c)*256 + s.  Destination K[s, :]
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


class AttnScoresKM256x36App(IpuApp):
    """kQᵀ → key-major scores, single head (D=36, N=256)."""

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

        # CR1 (≡1) is read-only hardwired; cr0 (=QBASE=0x0) matches hardwired 0.
        state.regfile.set_cr(0, QBASE_ROW)
        state.regfile.set_cr(2, SBASE_ROW)
        state.regfile.set_cr(9, KBASE_KM_ROW)
        state.regfile.set_cr(5, -Q_CHAN_ROWS)   # g=0 channel-column startup (rows)
        state.regfile.set_cr(6, -K_STRIDE_ROWS) # g=1 channel-column startup (rows)
        state.regfile.set_cr(7, -1)          # fixed_idx c startup
        state.regfile.set_cr(8, D - 2)       # c-loop bound: first=0, width=D → D-2 = 34

        state.regfile.set_lr(0, 0)           # R_CYCLIC index 0
        state.regfile.set_lr(2, Q_CHAN_ROWS)    # channel stride in Q (rows)
        state.regfile.set_lr(3, OUT_ROWS)       # output store stride (rows)
        state.regfile.set_lr(6, D - 2)       # c-loop bound = 34
        state.regfile.set_lr(7, 0)           # output byte pointer
        state.regfile.set_lr(8, -K_STRIDE_ROWS) # key row offset startup (-1 -> first live 0)
        state.regfile.set_lr(9, 0)           # key counter
        state.regfile.set_lr(10, N_TOK)      # key-loop limit
        state.regfile.set_lr(12, K_STRIDE_ROWS) # key stride into K scratch (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                SBASE, OUTPUT_ROW_BYTES, N_TOK * N_TG,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's exact-match shape domain
# (n_tok=256, d=36). `head` stays OUT of `requires`/`supports`, same reasoning
# as the other two attn_scores_km apps: it picks which of the 4 heads already
# in the input file to compute, not a shape dimension the registry routes on.
# Like its siblings, the constructor range-checks `head` against N_HEADS at
# construction time.


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
        f"n_tok == {N_TOK} and d == {D} exactly: the key-major kQT scores "
        f"kernel, one selected head of {N_HEADS} out of a 4-head input file, "
        f"256 tokens spanning {N_TG} 128-lane query groups."
    )


SPEC = KernelSpec(
    name="attn_scores_km_256x36",
    op="attn_scores_km",
    variant="256x36",
    app_class=AttnScoresKM256x36App,
    asm="attn_scores_km_256x36.asm",
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
