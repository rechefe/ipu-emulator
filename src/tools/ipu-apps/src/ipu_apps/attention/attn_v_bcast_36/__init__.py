"""attn@V (broadcast kernel) harness — key-major scores, channel-major output.

Computes, per attention head h in [0,4):
    O[i, t] = sum_s P[i, s] * V[s, t]      i, s in [0, 256),  t in [0, 36)

Lanes = query tokens, scalar = V[s,t] indexed from R0++R1, ACC accumulates over
keys — the standard matmul broadcast (no AGG, no collision). The companion
query-major + AGG kernel is `attn_v_256x36`; this app is the key-major variant
and shares its V and O byte layouts so the two are directly comparable.

Inputs (1 byte/element):
  P key-major  : P[i, s] at PBASE + h*65536 + s*256 + i   (4 heads, head-major)
  V channel-major: V[s, chan] at VBASE + chan*256 + s,  chan = h*36 + t

Output (FP32 R_ACC, 512-byte group rows — transformer-matmul convention):
  O[i, t] at OBASE + chan*1024 + g*512 + local*4,  i = g*128 + local
  i.e. channel (h*36 + t) occupies 1024 bytes (2 groups of 128 FP32 lanes).

Usage::

    from ipu_apps.attention.attn_v_bcast_36 import AttnVBcast36App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, ShapeBundle, no, yes
from ipu_apps.attention._spec_support import attn_v_bcast_query, positive_dims

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
PV_STRIDE_ROWS     = N_TOK // LANES          # 2: rows per P key / V channel
R1_OFF_ROWS        = 1                       # R1 source is the next row
P_HEAD_STRIDE_ROWS = N_TOK * PV_STRIDE_ROWS  # 512: rows per head in P
OUT_ROW_ROWS       = 1                       # one r_acc store = one row (wide)
GRP_QUERY_ROWS     = 1                       # group query offset
O_CHAN_ROWS        = 2 * OUT_ROW_ROWS        # 2 groups per output channel

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


class AttnVBcast36App(IpuApp):
    """attn@V broadcast kernel harness (4 heads, N=256, head_dim=36)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_path = Path(self.p_path)
        self.v_path = Path(self.v_path)

    def setup(self, state: "IpuState") -> None:
        # P and V are stored verbatim (already in the kernel's byte layout).
        state.xmem.write_address(PBASE, bytearray(self.p_path.read_bytes()))
        state.xmem.write_address(VBASE, bytearray(self.v_path.read_bytes()))

        # CR0 (==0) and CR1 (==1) are hardwired; the rest are writable.
        state.regfile.set_cr(2, PBASE_ROW)
        state.regfile.set_cr(3, VBASE_ROW)
        state.regfile.set_cr(4, OBASE_ROW)
        state.regfile.set_cr(5, R1_OFF_ROWS)        # R1 source offset within V channel (rows)
        state.regfile.set_cr(6, -1)               # key-index startup
        state.regfile.set_cr(7, P_HEAD_STRIDE_ROWS) # P head stride (rows)
        # Loop bounds are count-1: the counter ADD and the BLT share one bundle,
        # so BLT reads the pre-ADD snapshot (branch taken while snapshot < bound).
        state.regfile.set_cr(8, N_TOK - 2)        # 254: key-loop bound (width 256, peeled+startup)
        state.regfile.set_cr(9, D - 1)            # 35: t-loop bound (36 channels)
        state.regfile.set_cr(10, N_HEAD - 1)      # 3: head-loop bound (4 heads)
        state.regfile.set_cr(11, 1)               # 1: g-loop bound (2 groups)
        # LRs
        state.regfile.set_lr(0, 0)                # r_cyclic index / mask_shift
        state.regfile.set_lr(1, PV_STRIDE_ROWS)     # P key stride / V channel stride (rows)
        state.regfile.set_lr(2, OUT_ROW_ROWS)       # output-row stride (rows)
        state.regfile.set_lr(3, GRP_QUERY_ROWS)     # group query offset (rows)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            # 144 channels, each 1024 B (two 512-B FP32 group rows).
            dump_xmem_to_binary(
                state, self.output_path,
                OBASE, O_CHAN_BYTES, N_CHAN,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. Unlike the
# other three attention ops, attn_v_bcast is indexed by `d` (head_dim) ALONE:
# its three sibling apps are named and distinguished purely by head_dim
# (36/48/60), and N_TOK is a fixed module constant that is not part of the
# dirname or a caller-visible constructor parameter -- there is nothing for a
# query to assert it against, so `n_tok` is deliberately absent from `requires`
# for this op (see ``_spec_support.AttnVBcastQuery``).


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
        f"d == {D} exactly: the attn@V broadcast kernel (key-major P, "
        f"ACC.ADD, no AGG), fixed at n_tok={N_TOK}, all {N_HEAD} heads in one "
        f"invocation."
    )


SPEC = KernelSpec(
    name="attn_v_bcast_36",
    op="attn_v_bcast",
    variant="36",
    app_class=AttnVBcast36App,
    asm="attn_v_bcast_36.asm",
    requires=("d",),
    tags=("fp32-wide", "key-major"),
    supports=_supports,
    build=_build,
    explain=_explain,
    bundle=lambda **params: ShapeBundle.of(
        attn=(N_TOK, N_TOK), value=(N_TOK, params["d"])
    ).with_shapes(derived={"output": (N_TOK, params["d"])}),
    # Exact-shape match: no padding, no chunking. Cheapest possible claim.
    cost=lambda **params: 0.0,
)
