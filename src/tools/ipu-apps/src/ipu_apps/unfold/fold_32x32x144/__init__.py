"""Fold 32x32x144: 4 channel-major streams -> spatial 32x32x144 (L3).

Exact inverse of :mod:`ipu_apps.unfold.unfold_32x32x144`: takes the same 4
channel-major streams (TL, TR, BL, BR) that kernel produces -- each stream
split into 2 token groups (tg0, tg1) per channel -- and reconstructs the
original NHCW-striped spatial tensor, in the same 8-stripe x 144-channel x
128-element layout unfold's own input used.

Each stream row (one per (channel, token-group) pair) carries a FULL 128
valid FP32 tokens -- unlike fold_16x16x192, there is no stale padding tail to
ignore here; every lane of every loaded stream row is real data (see the
``.asm`` header for the exact ACC.RESHAPE-based inverse mapping and how the
(stream, tg) pair for a destination stripe was derived empirically).

Usage::

    from ipu_apps.unfold.fold_32x32x144 import Fold32x32x144App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import KernelSpec, no, yes
from ipu_apps.unfold._spec_support import (
    WIDE_VECTOR_ONLY,
    positive_dims,
    unfold_query,
)

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions (shared with unfold_32x32x144; same (H, W, C) triple) -------

H         = 32    # spatial height
W         = 32    # spatial width
C         = 144   # channels
N_STRIPES = 8     # 128-element row = 4 spatial rows x 32 cols -> H/4 = 8 stripes
N_STREAMS = 4     # TL, TR, BL, BR
N_TG      = 2     # token groups per channel (unfold_32x32x144's stream rows)
N_TOK     = 128   # valid tokens per stream row -- a FULL 128-lane row (no padding)

# -- Memory map ---------------------------------------------------------
#
# Wide-vector FP32 only, same convention as fold_16x16x192: an XMEM row is
# LANES * 4 = 512 bytes unconditionally, and .asm XMEM operands are ROW
# numbers (issue #179), not byte offsets.
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

DST_ROW_BYTES = ROW_BYTES                    # 512 bytes/row in wide-vector debug mode

_STREAM_ROWS = C * N_TG                      # rows per input stream (2 tgs per channel)
SRC_STRIDE_ROWS = N_TG                       # one src (ch,tg0/tg1) pair per channel step
DST_STRIDE_ROWS = 1                          # one dst row per channel, per stripe

SRC_BASE_ROW = 0
DST_BASE_ROW = SRC_BASE_ROW + N_STREAMS * _STREAM_ROWS
ONES_BASE_ROW = DST_BASE_ROW + N_STRIPES * C

SRC_BASE  = SRC_BASE_ROW * ROW_BYTES
DST_BASE  = DST_BASE_ROW * ROW_BYTES
ONES_BASE = ONES_BASE_ROW * ROW_BYTES


# -- XMEM loaders -------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write the 4 channel-major streams directly into XMEM at SRC_BASE.

    File layout: (4 streams x 144 channels x 2 token-groups) rows, each 512
    bytes (matches unfold_32x32x144's OUTPUT layout exactly). Row (stream,
    ch, tg) at offset (stream * 288 + ch * 2 + tg) * 512. Every lane of every
    row is valid data (no stale-padding tail, unlike fold_16x16x192).
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState") -> None:
    """One XMEM row of FP32 1.0 for r_cyclic (the pass-through multiplier)."""
    state.xmem.write_address(ONES_BASE, bytearray(np.ones(LANES, dtype=np.float32).tobytes()))


def _pack_bytes(values: list[int]) -> int:
    """Pack 4 small integers (each in [0, 255]) into one little-endian 32-bit word.

    Used to build ACC.RESHAPE index tables in a CR so ``SET`` can copy them
    into an LR: the CR's 4 raw bytes become an LRDn pair's 4 index elements
    once two such CRs are SET into the two halves of the pair.
    """
    assert len(values) == 4
    return int.from_bytes(bytes(v & 0xFF for v in values), "little")


# -- App ------------------------------------------------------------------

class Fold32x32x144App(IpuApp):
    """Fold 4 channel-major streams (TL, TR, BL, BR) into a 32x32x144 spatial tensor.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to the 4-stream input (4 x 288 x 512 bytes; matches
            unfold_32x32x144's output).
        output_path: Optional path to write the NHCW-striped spatial output
            (matches unfold_32x32x144's input).
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)

        # cr13: DST_BASE_ROW (stripe-0 output base). CR0 and CR1 are BOTH
        # read-only hardwired constants (0 and 1 respectively; writes are
        # silently dropped -- see CR_READ_ONLY_INITIAL_VALUES). DST_BASE_ROW
        # is nonzero (it sits after the 4-stream source region), so it
        # cannot use cr0 -- same trap fold_16x16x192 documents. cr13 is used
        # instead (cr14 is free too; either works, cr13 chosen arbitrarily).
        state.regfile.set_cr(13, DST_BASE_ROW)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # cr9..cr12: per-stream SOURCE bases (TL, TR, BL, BR)
        state.regfile.set_cr(9,  SRC_BASE_ROW)
        state.regfile.set_cr(10, SRC_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, SRC_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, SRC_BASE_ROW + 3 * _STREAM_ROWS)

        # cr2/cr3: source-lane-index table for ACC.RESHAPE, packed as two
        # 4-byte halves of an LRDn pair -- [0,1,2,3] then [4,5,6,7]. Rebuilt
        # via SET at the start of every stream's 4-call block in the .asm
        # (never assumed preloaded in any LR).
        state.regfile.set_cr(2, _pack_bytes([0, 1, 2, 3]))
        state.regfile.set_cr(3, _pack_bytes([4, 5, 6, 7]))
        # cr4/cr5: stream-TL destination-lane-index table for call 0, packed
        # the same way -- [0,2,4,6] then [8,10,12,14]. TR/BL/BR are derived
        # from this via ADDBI (+1/+32/+33) in the .asm. This table (and its
        # per-call step sequence +16/+48/+16) is IDENTICAL across all 8
        # stripes -- only the source-side window base and the tg selection
        # change per stripe (see the .asm header derivation).
        state.regfile.set_cr(4, _pack_bytes([0, 2, 4, 6]))
        state.regfile.set_cr(5, _pack_bytes([8, 10, 12, 14]))

        # constant LRs preset here (SET requires a CR source since issue #82;
        # set_lr in the harness has no such restriction).
        #
        # STR_POST_AAQ_REG's base operand must be a CR, not an LR (discovered
        # when a first draft tried to pass a running LR as the store base and
        # the assembler rejected it -- "CrRegField" only accepts cr0..cr15).
        # So the destination row address is CR13 (DST_BASE_ROW, fixed) + LR8
        # (running offset), and LR8 accumulates the FULL stripe*C + ch offset
        # across the ENTIRE kernel (0 .. N_STRIPES*C - 1) rather than
        # resetting to 0 at each stripe boundary -- there is no per-stripe CR
        # to add a reset value back onto.
        state.regfile.set_lr(0, 0)      # r_cyclic slot 0
        state.regfile.set_lr(6, 0)      # src row offset within a stream = ch*N_TG+tg; reset every stripe
        state.regfile.set_lr(7, SRC_STRIDE_ROWS)   # src stride per channel (2 rows: tg0, tg1)
        state.regfile.set_lr(8, 0)      # dst row offset = stripe*C + ch (relative to DST_BASE_ROW); += 1 per channel, never reset
        state.regfile.set_lr(9, DST_STRIDE_ROWS)   # dst stride per channel (1 row)
        state.regfile.set_lr(10, 0)     # channel counter (inner loop); reset every stripe
        state.regfile.set_lr(11, C)     # inner loop limit

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                DST_BASE, DST_ROW_BYTES, N_STRIPES * C,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: it is an exact-shape
# match, since the stripe/packing geometry is baked into the .asm for this one
# (H, W, C) triple -- same convention unfold_32x32x144 and fold_16x16x192 use.


def _supports(**params):
    q = unfold_query(params["shape"])
    bad = positive_dims(q)
    if bad:
        return no(bad)
    if (q.h, q.w, q.c) != (H, W, C):
        return no(
            f"handles exactly (H, W, C) = ({H}, {W}, {C}); got ({q.h}, {q.w}, {q.c})"
        )
    return yes()


def _build(**params):
    return {}


def _explain(**params):
    return (
        f"(H, W, C) == ({H}, {W}, {C}) exactly: geometry (stripe count, spatial "
        f"row packing, register layout) is fixed in the .asm for this shape."
    )


SPEC = KernelSpec(
    name="fold_32x32x144",
    op="fold",
    variant="32x32x144",
    app_class=Fold32x32x144App,
    asm="fold_32x32x144.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: unfold_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
