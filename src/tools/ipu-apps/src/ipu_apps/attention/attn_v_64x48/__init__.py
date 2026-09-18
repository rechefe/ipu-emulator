"""attn@V (AGG kernel), Layer 4 — query-major scores, channel-major output.

Computes, per (stream, head) block b of Layer 4::

    O[b, i, t] = sum_s P[b, i, s] * V[b, s, t]    i, s in [0, 64), t in [0, 48)

Layer 4 parameters: d = 192, N = 64 tokens per stream, P = 4 streams,
h = 4 heads, head_dim = 48.  There are ``P * N_HEAD = 16`` (stream, head)
blocks and ``N_BLOCK * D = 768`` value channels overall.

This is the L4 port of ``attn_v_256x36``.  The mapping is carried over
unchanged: lanes are KEYS, ``MULT.RC.VV`` forms the element-wise product of a
query's score row (R0) with a value channel's key column (R_CYCLIC), and
``AGG.SUM[.FIRST]`` reduces the live MULT_RES lanes into one R_ACC slot per
query -- collision-free, no ACC.

The structural difference from L3 is that N = 64 <= LANES = 128:
  * A key axis fits in ONE row, so there is a single key chunk per channel --
    the L3 chunk-0/chunk-1 split collapses and every AGG is an ``AGG.SUM.FIRST``
    (one clean write, no cross-chunk accumulation).
  * A query group is 64 queries, so one R_ACC store covers a whole channel and
    there is a single output group per channel. AGG dest slots run 0..63 and
    R_ACC lanes 64..127 are unused padding, cropped in ``teardown``.
Rows are never shared: each output channel owns a whole 512 B row.

AGG.SUM.FIRST reduces across the lane (key) axis, so the trailing 64 lanes of
P and V must be excluded structurally rather than relied on to be zero: a real
producer in a chained pipeline would leave another stream's data there.
``setup`` sets ``valid_elements = N_TOK`` on the dstructure CR so the AGG
reduction only ever sees the 64 live key lanes.

It consumes QUERY-major scores, so it pairs with ``qk_scores_64x48``. The
key-major chain is ``attn_scores_km_64x48`` + ``attn_v_bcast_48``; the two
chains produce bit-different results by design and must not be mixed.

Usage::

    from ipu_apps.attention.attn_v_64x48 import AttnV64x48App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, ShapeBundle, no, yes
from ipu_apps.attention._spec_support import attn_v_query, positive_dims

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

N_TOK   = 64                  # queries == keys, per stream
D       = 48                  # head_dim
P       = 4                   # streams / partitions
N_HEAD  = 4                   # attention heads
N_BLOCK = P * N_HEAD          # 16 independent (stream, head) blocks
N_CHAN  = N_BLOCK * D         # 768 value channels total

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
# N_TOK = 64 <= LANES, so a query's score row and a value channel's key column
# are each ONE whole row (trailing lanes unused).
PV_STRIDE_ROWS     = 1                       # rows per P query row / V channel
P_BLOCK_ROWS       = N_TOK * PV_STRIDE_ROWS  # 64: P rows per block
V_BLOCK_ROWS       = D * PV_STRIDE_ROWS      # 48: V rows per block
O_CHAN_ROWS        = 1                       # one r_acc store = one row (wide)

P_ROWS = N_BLOCK * P_BLOCK_ROWS              # 1024
V_ROWS = N_CHAN * PV_STRIDE_ROWS             # 768
O_ROWS = N_CHAN * O_CHAN_ROWS                # 768

PBASE_ROW = 0
VBASE_ROW = PBASE_ROW + P_ROWS
OBASE_ROW = VBASE_ROW + V_ROWS

PBASE = PBASE_ROW * ROW_BYTES
VBASE = VBASE_ROW * ROW_BYTES
OBASE = OBASE_ROW * ROW_BYTES

# One store = one whole row; only the first N_TOK lanes hold queries.
OUTPUT_ROW_BYTES = N_TOK * ELEM_BYTES        # 256 valid bytes per channel row


class AttnV64x48App(IpuApp):
    """Layer-4 attn@V AGG kernel harness (16 blocks, N=64, head_dim=48)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are staged verbatim (already in the kernel's row layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # AGG.SUM.FIRST reduces across the lane (key) axis, so the trailing
        # 64 lanes must be excluded structurally: a chained producer would
        # leave real data there, not zeros.
        state.set_cr_dstructure(valid_elements=N_TOK)

        # CR0 (==0) and CR1 (==1) are hardwired read-only, so every base lives
        # on a writable CR -- PBASE_ROW happening to be 0 must not be relied on.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(5, PV_STRIDE_ROWS)     # P query stride (rows)
        state.regfile.set_cr(8, P_BLOCK_ROWS)       # P rows per block (64)
        state.regfile.set_cr(9, N_TOK - 1)          # 63: inner bound -> 64 AGG bundles
        state.regfile.set_cr(10, D)                 # 48: channels t per block
        state.regfile.set_cr(11, N_BLOCK)           # 16: (stream, head) blocks
        state.regfile.set_cr(12, O_CHAN_ROWS)       # O channel stride (rows)

    def teardown(self, state: "IpuState") -> None:
        """Crop each channel's valid N_TOK queries out of its whole 512 B row.

        Every store wrote a full row (one output channel per row -- rows are
        never shared), but only the leading ``N_TOK * ELEM_BYTES`` bytes hold
        results. The output file is the densely packed crop: N_CHAN rows of
        N_TOK FP32, channel (b*D + t) at row index b*D + t.
        """
        if self.output_path is None:
            return
        stride = O_CHAN_ROWS * ROW_BYTES
        parts = [
            bytes(state.xmem.read_address(OBASE + c * stride, OUTPUT_ROW_BYTES))
            for c in range(N_CHAN)
        ]
        Path(self.output_path).write_bytes(b"".join(parts))


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's exact-match shape domain
# (n_tok=64, d=48).


def _supports(**params):
    q = attn_v_query(n_tok=params["n_tok"], d=params["d"])
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
        f"n_tok == {N_TOK} and d == {D} exactly: the L4 attn@V AGG kernel "
        f"(query-major P), all {N_BLOCK} (stream, head) blocks in one "
        f"invocation."
    )


SPEC = KernelSpec(
    name="attn_v_64x48",
    op="attn_v",
    variant="64x48",
    app_class=AttnV64x48App,
    asm="attn_v_64x48.asm",
    requires=("n_tok", "d"),
    tags=("fp32-wide", "query-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        attn=(N_BLOCK, params["n_tok"], params["n_tok"]),
        value=(N_BLOCK, params["n_tok"], params["d"]),
    ).with_shapes(derived={"output": (N_BLOCK, params["n_tok"], params["d"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
