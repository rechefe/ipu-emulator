"""attn@V (broadcast kernel), Layer 4 — key-major scores, channel-major output.

Computes, per (stream, head) block b of Layer 4::

    O[b, i, t] = sum_s P[b, i, s] * V[b, s, t]    i, s in [0, 64), t in [0, 48)

Layer 4 parameters: d = 192, N = 64 tokens per stream, P = 4 streams,
h = 4 heads, head_dim = 48.  There are ``P * N_HEAD = 16`` (stream, head)
blocks and ``N_BLOCK * D = 768`` value channels overall.

This is the L4 port of ``attn_v_bcast_36``.  The mapping is carried over
unchanged: lanes are QUERIES, the scores are KEY-major so one key's 64 query
scores are one R_CYCLIC row, ``V[s, t]`` is the broadcast scalar taken from R0,
and ``ACC.ADD[.FIRST]`` accumulates over the 64 keys.  This is the standard
matmul-broadcast template -- **no AGG**, no collision, so the golden is a plain
float32 accumulation, not an AGG float64 left-fold.

The structural differences from L3 are both consequences of N = 64 <= LANES:
  * A query axis fits in ONE row, so the L3 g=0/g=1 query-group split collapses
    to a single group: one ACC chain and one store per (block, channel).
  * All 64 keys of a value channel fit in R0 alone, so the L3 R0++R1 pair
    (which spanned 256 keys) collapses to a single ``LDR_MULT_REG r0``.
Rows are never shared: each output channel owns a whole 512 B row and the
teardown crops the valid ``N_TOK * ELEM_BYTES`` prefix.

No AGG means no cross-lane reduction: ``ACC.ADD`` accumulates each of the 128
lanes (queries) independently, so the 64 padding lanes can only ever waste
lanes -- they cannot contaminate the 64 valid ones regardless of their
content. (Contrast ``attn_v_64x48``, whose ``AGG.SUM.FIRST`` reduces across
lanes and therefore needs ``valid_elements`` to exclude padding structurally.)
The test proves this with a garbage-padding probe rather than relying on the
harness's zero-fill.

V and O use exactly the layouts of the query-major sibling ``attn_v_64x48`` --
V channel-major on row ``b*D + t`` and O channel-major on row ``b*D + t``, both
cropped to N_TOK FP32 -- so the two attn@V variants are directly comparable.
Only P differs: this kernel consumes KEY-major scores, so it pairs ONLY with
``attn_scores_km_64x48``, never with ``qk_scores_64x48``.  The key-major and
query-major chains produce bit-different results by design.

Usage::

    from ipu_apps.attention.attn_v_bcast_48 import AttnVBcast48App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, ShapeBundle, no, yes
from ipu_apps.attention._spec_support import attn_v_bcast_query, positive_dims

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
# N_TOK = 64 <= LANES, so a key's score row and a value channel's key column
# are each ONE whole row (trailing lanes unused).
PV_STRIDE_ROWS = 1                           # rows per P key row / V channel
P_BLOCK_ROWS   = N_TOK * PV_STRIDE_ROWS      # 64: P rows per block
V_BLOCK_ROWS   = D * PV_STRIDE_ROWS          # 48: V rows per block
O_CHAN_ROWS    = 1                           # one r_acc store = one row (wide)

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


class AttnVBcast48App(IpuApp):
    """Layer-4 attn@V broadcast kernel harness (16 blocks, N=64, head_dim=48)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are staged verbatim (already in the kernel's row layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # CR0 (==0) and CR1 (==1) are hardwired read-only, so every base lives
        # on a writable CR -- PBASE_ROW happening to be 0 must not be relied on.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(5, -PV_STRIDE_ROWS)    # -1 row: P key startup skew
        state.regfile.set_cr(6, -1)                 # key fixed_idx startup
        state.regfile.set_cr(7, P_BLOCK_ROWS)       # 64: P rows per block
        # Loop bound is width-2: the load runs a bundle ahead of the MULT that
        # consumes it, and the .asm raises this by one in-kernel for the BLT.
        state.regfile.set_cr(8, N_TOK - 2)          # 62: key-loop bound (width 64)
        state.regfile.set_cr(10, D)                 # 48: channels t per block
        state.regfile.set_cr(11, N_BLOCK)           # 16: (stream, head) blocks
        state.regfile.set_cr(12, O_CHAN_ROWS)       # 1: V/O channel stride (rows)

        state.regfile.set_lr(0, 0)                  # r_cyclic index / mask_shift
        state.regfile.set_lr(1, PV_STRIDE_ROWS)     # P key stride (rows)

    def teardown(self, state: "IpuState") -> None:
        """Crop each channel's valid N_TOK queries out of its whole 512 B row.

        Every store wrote a full row (one output channel per row -- rows are
        never shared), but only the leading ``N_TOK * ELEM_BYTES`` bytes hold
        results. The output file is the densely packed crop: N_CHAN rows of
        N_TOK FP32, channel (b*D + t) at row index b*D + t -- byte-identical in
        shape to ``attn_v_64x48``'s output.
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
# Declared beside the kernel so the registry needs no central list. Like its
# siblings, attn_v_bcast is indexed by `d` (head_dim) ALONE -- N_TOK is a
# fixed module constant, not a caller-visible parameter, so it is absent from
# `requires` (see ``_spec_support.AttnVBcastQuery``).


def _supports(**params):
    q = attn_v_bcast_query(d=params["d"])
    bad = positive_dims(d=q.d)
    if bad:
        return no(bad)
    if q.d != D:
        return no(f"handles exactly d={D}; got {q.d}")
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"d == {D} exactly: the L4 attn@V broadcast kernel (key-major P, "
        f"ACC.ADD, no AGG), fixed at n_tok={N_TOK}, all {N_BLOCK} (stream, "
        f"head) blocks in one invocation."
    )


SPEC = KernelSpec(
    name="attn_v_bcast_48",
    op="attn_v_bcast",
    variant="48",
    app_class=AttnVBcast48App,
    asm="attn_v_bcast_48.asm",
    requires=("d",),
    tags=("fp32-wide", "key-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        attn=(N_BLOCK, N_TOK, N_TOK), value=(N_BLOCK, N_TOK, params["d"])
    ).with_shapes(derived={"output": (N_BLOCK, N_TOK, params["d"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
