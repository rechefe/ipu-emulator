"""Unfold 16×16×192 → 4 channel-major streams (L4).

Rearranges a 16×16×192 spatial tensor (NHCW striped) into four 8×8
sub-grid streams (TL, TR, BL, BR), each output in channel-major FP32 format.

NOT a port of :mod:`ipu_apps.unfold.unfold_32x32x144` — the geometry differs. L3 has
8 stripes of 4 spatial rows × 32 cols; L4 has 2 stripes of 8 rows × 16 cols.
A stream therefore fills only 2 of the 4 ``ACC.STRIDE`` slots (64 tokens), so
each 512-byte output row carries **64 valid FP32 tokens (256 bytes) followed by
256 bytes of stale r_acc lanes**. ``STR_ACC_REG`` always writes the full
512-byte register (``ipu.py:446-455``), and consumers must read only the first
half of each row. This is the deliberate per-stream layout (plan §4); the
packed ``[k][p·n]`` variant is the deferred §9 experiment.

Usage::

    from ipu_apps.unfold.unfold_16x16x192 import Unfold16x16x192App
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

# -- Dimensions -------------------------------------------------------------

H         = 16    # spatial height
W         = 16    # spatial width
C         = 192   # channels
N_STRIPES = 2     # 128-byte row = 8 spatial rows × 16 cols → H/8 = 2 stripes
N_STREAMS = 4     # TL, TR, BL, BR
N_OUT     = C     # output channels per stream (same as input channels)
N_TOK     = 64    # tokens per stream (8×8 sub-grid) — HALF of a 128-lane row

# -- Memory map -------------------------------------------------------------

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

OUTPUT_ROW_BYTES = 512                       # one r_acc store = one row in wide mode
VALID_ROW_BYTES  = N_TOK * ELEM_BYTES        # meaningful prefix of each output row

_STRIPE_ROWS = C                             # one row per channel within a stripe
_STREAM_ROWS = N_OUT * 1               # rows per output stream
SRC_STRIDE_ROWS = 1                          # one src row per channel
DST_STRIDE_ROWS = 1                    # rows per output channel

SRC_BASE_ROW  = 0
ONES_BASE_ROW = SRC_BASE_ROW + N_STRIPES * _STRIPE_ROWS
DST_BASE_ROW  = ONES_BASE_ROW + 1

SRC_BASE  = SRC_BASE_ROW * ROW_BYTES
ONES_BASE = ONES_BASE_ROW * ROW_BYTES
DST_BASE  = DST_BASE_ROW * ROW_BYTES


# -- XMEM loaders -----------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write NHCW-striped input directly into XMEM at SRC_BASE.

    File layout: (2 stripes × 192 channels) rows, each 128 bytes.
    Row (stripe, ch) at offset (stripe × 192 + ch) × 128.
    Each row: 8 spatial rows × 16 columns of one channel.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState") -> None:
    """One XMEM row of FP32 1.0 for r_cyclic (the pass-through multiplier)."""
    state.xmem.write_address(ONES_BASE, bytearray(np.ones(LANES, dtype=np.float32).tobytes()))


# -- App --------------------------------------------------------------------

class Unfold16x16x192App(IpuApp):
    """Unfold a 16×16×192 spatial tensor into 4 channel-major streams.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to NHCW-striped input (49,152 bytes).
        output_path: Optional path to write the 4-stream FP32 output.
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)
        # cr0, cr13: per-stripe source bases. CR1 (≡1) is a read-only hardwired
        # constant — writes are silently dropped — so stripe 1 goes to CR13.
        # cr0 = SRC_BASE + 0 is 0x0 (harmless no-op, matches hardwired 0).
        state.regfile.set_cr(0, SRC_BASE_ROW)
        state.regfile.set_cr(13, SRC_BASE_ROW + _STRIPE_ROWS)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # cr9..cr12: per-stream destination bases (TL, TR, BL, BR)
        state.regfile.set_cr(9,  DST_BASE_ROW)
        state.regfile.set_cr(10, DST_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, DST_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, DST_BASE_ROW + 3 * _STREAM_ROWS)
        # constant LRs preset here (SET requires a CR source since issue #82)
        state.regfile.set_lr(0, 0)      # r_cyclic slot 0 / mask / acc.stride slot 0
        state.regfile.set_lr(1, 1)      # acc.stride r_acc slot 1 → [32..63]
        state.regfile.set_lr(4, 0)      # src byte offset, += 128 per channel
        state.regfile.set_lr(5, SRC_STRIDE_ROWS)   # src stride per channel (1 row)
        state.regfile.set_lr(6, DST_STRIDE_ROWS)   # dst stride per channel (4 rows)
        state.regfile.set_lr(8, 0)      # dst byte offset = ch × 512
        state.regfile.set_lr(10, 0)     # channel counter
        state.regfile.set_lr(11, C)     # loop limit

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                DST_BASE, OUTPUT_ROW_BYTES, N_STREAMS * N_OUT,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: it is an exact-shape
# match, since the stripe/packing geometry is baked into the .asm for this one
# (H, W, C) triple.


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
    name="unfold_16x16x192",
    op="unfold",
    variant="16x16x192",
    app_class=Unfold16x16x192App,
    asm="unfold_16x16x192.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: unfold_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
