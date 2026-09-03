"""Fold 16x16x192: 4 channel-major streams -> spatial 16x16x192 (L4).

Exact inverse of :mod:`ipu_apps.unfold.unfold_16x16x192`: takes the same 4
channel-major streams (TL, TR, BL, BR) that kernel produces and reconstructs
the original NHCW-striped spatial tensor, in the same 2-stripe x 192-channel
x 128-element layout unfold's own input used.

Each stream row carries only 64 valid FP32 tokens (256 bytes); lanes 64..127
are unfold's stale r_acc padding and this kernel never reads them (see the
``.asm`` header for the exact byte layout and the ACC.RESHAPE-based inverse
mapping).

Usage::

    from ipu_apps.unfold.fold_16x16x192 import Fold16x16x192App
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

# -- Dimensions (shared with unfold_16x16x192; same (H, W, C) triple) -------

H         = 16    # spatial height
W         = 16    # spatial width
C         = 192   # channels
N_STRIPES = 2     # 128-element row = 8 spatial rows x 16 cols -> H/8 = 2 stripes
N_STREAMS = 4     # TL, TR, BL, BR
N_TOK     = 64    # valid tokens per stream row (8x8 sub-grid) -- HALF of a 128-lane row

# -- Memory map ---------------------------------------------------------
#
# Wide-vector FP32 only, same convention as unfold_16x16x192: an XMEM row is
# LANES * 4 = 512 bytes unconditionally, and .asm XMEM operands are ROW
# numbers (issue #179), not byte offsets.
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

OUTPUT_ROW_BYTES = 128 * ELEM_BYTES          # narrow-mode row size (128 elements = 512B here too in wide mode)
# Fold's output is the spatial tensor in unfold's INPUT layout: each row is
# only 128 elements of *narrow* data conceptually, but this kernel runs in
# wide-vector FP32 debug mode, so a row occupies a full 512-byte XMEM row
# just like unfold_16x16x192's input did.
DST_ROW_BYTES = ROW_BYTES                    # 512 bytes/row in wide-vector debug mode

_STREAM_ROWS = C                             # rows per input stream (one per channel)
SRC_STRIDE_ROWS = 1                          # one src row per channel, per stream
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

    File layout: (4 streams x 192 channels) rows, each 512 bytes (matches
    unfold_16x16x192's OUTPUT layout exactly). Row (stream, ch) at offset
    (stream * 192 + ch) * 512. Only the first 256 bytes (64 FP32 tokens) of
    each row are meaningful; the trailing 256 bytes are ignored.
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

class Fold16x16x192App(IpuApp):
    """Fold 4 channel-major streams (TL, TR, BL, BR) into a 16x16x192 spatial tensor.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to the 4-stream input (4 x 192 x 512 bytes; matches
            unfold_16x16x192's output).
        output_path: Optional path to write the NHCW-striped spatial output
            (matches unfold_16x16x192's input).
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)

        # cr14, cr13: per-stripe DESTINATION bases. CR0 and CR1 are BOTH
        # read-only hardwired constants (0 and 1 respectively; writes are
        # silently dropped -- see CR_READ_ONLY_INITIAL_VALUES). unfold's own
        # cr0 usage for its stripe-0 SOURCE base only works because that
        # base happens to be 0; fold's stripe-0 DESTINATION base is
        # DST_BASE_ROW (nonzero), so cr14 is used instead of cr0.
        state.regfile.set_cr(14, DST_BASE_ROW)
        state.regfile.set_cr(13, DST_BASE_ROW + C)
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
        # from this via ADDBI (+1/+16/+17) in the .asm.
        state.regfile.set_cr(4, _pack_bytes([0, 2, 4, 6]))
        state.regfile.set_cr(5, _pack_bytes([8, 10, 12, 14]))

        # constant LRs preset here (SET requires a CR source since issue #82)
        state.regfile.set_lr(0, 0)      # r_cyclic slot 0
        state.regfile.set_lr(6, 0)      # src row offset within a stream; += 1 per channel
        state.regfile.set_lr(7, SRC_STRIDE_ROWS)   # src stride per channel (1 row)
        state.regfile.set_lr(8, 0)      # dst row offset = ch; += 1 per channel
        state.regfile.set_lr(9, DST_STRIDE_ROWS)   # dst stride per channel (1 row)
        state.regfile.set_lr(10, 0)     # channel counter
        state.regfile.set_lr(11, C)     # loop limit

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
# (H, W, C) triple -- same convention unfold_16x16x192 uses.


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
    name="fold_16x16x192",
    op="fold",
    variant="16x16x192",
    app_class=Fold16x16x192App,
    asm="fold_16x16x192.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: unfold_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
