"""Fold 8x8x240: 4 stride-2 channel-major streams -> spatial 8x8x240 (L5).

Exact inverse of :mod:`ipu_apps.unfold.unfold_8x8x240`: takes the same 4
stride-2 decimated streams that kernel produces and reconstructs the
original 8x8x240 spatial tensor, ONE channel per row, in NAIVE (unpacked)
row-major spatial order -- lane = row*8 + col.

This is a deliberate departure from L3/L4's convention of matching fold's
output byte-for-byte to unfold's own INPUT layout: unfold_8x8x240's INPUT
requires the caller to pre-permute spatial rows via
``unfold_8x8x240._ROW_PACK_ORDER`` (a workaround for W=8 not being an
encodable ``elements_in_row``). ``ACC.RESHAPE``'s dest indices are arbitrary
bytes in [0, 127] -- unlike ``ACC.STRIDE``'s fixed view-row/view-col
structure -- so fold's scatter can target TRUE spatial (row, col) positions
directly, with NO equivalent output-side packing step. This was verified
empirically before writing the ``.asm`` (see its header): both the packed
and naive dest-index tables partition the 64 valid lanes cleanly with a
uniform per-call stride, but the naive layout is the more useful and more
natural output contract (a genuine ``[C, H, W]`` tensor), so it is what
this kernel produces. Fold's SOURCE-side interpretation still has to account
for ``_ROW_PACK_ORDER`` correctly -- but it does so implicitly, because it
consumes unfold's OUTPUT (already real-coordinate-ordered by unfold's own
(h, v) stride selectors), not unfold's packed INPUT rows.

Usage::

    from ipu_apps.unfold.fold_8x8x240 import Fold8x8x240App
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

# -- Dimensions (shared with unfold_8x8x240; same (H, W, C) triple) ---------

H         = 8     # spatial height
W         = 8     # spatial width
C         = 240   # channels
N_STREAMS = 4     # four stride-2 phases
N_TOK     = 16    # valid tokens per stream row (4x4 decimated grid)
N_VALID   = H * W # valid tokens per destination row (64) -- naive spatial layout

# -- Memory map ---------------------------------------------------------
#
# Wide-vector FP32 only, same convention as fold_16x16x192/fold_32x32x144:
# an XMEM row is LANES * 4 = 512 bytes unconditionally, and .asm XMEM
# operands are ROW numbers (issue #179), not byte offsets.
ELEM_BYTES = 4                               # FP32
LANES      = 128                             # elements per XMEM row
ROW_BYTES  = LANES * ELEM_BYTES              # 512

DST_ROW_BYTES = ROW_BYTES                    # 512 bytes/row in wide-vector debug mode

_STREAM_ROWS = C                             # rows per input stream (one per channel)
SRC_STRIDE_ROWS = 1                          # one src row per channel, per stream
DST_STRIDE_ROWS = 1                          # one dst row per channel

SRC_BASE_ROW = 0
DST_BASE_ROW = SRC_BASE_ROW + N_STREAMS * _STREAM_ROWS
ONES_BASE_ROW = DST_BASE_ROW + C

SRC_BASE  = SRC_BASE_ROW * ROW_BYTES
DST_BASE  = DST_BASE_ROW * ROW_BYTES
ONES_BASE = ONES_BASE_ROW * ROW_BYTES


# -- XMEM loaders -------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write the 4 stride-2 streams directly into XMEM at SRC_BASE.

    File layout: (4 streams x 240 channels) rows, each 512 bytes (matches
    unfold_8x8x240's raw, UNCROPPED row output -- see teardown()'s
    ``.rows.bin`` sibling file in that kernel). Row (stream, ch) at offset
    (stream * 240 + ch) * 512. Only the first 16 FP32 lanes (64 bytes) of
    each row are meaningful; the trailing 112 lanes are unfold's stale
    r_acc padding and this kernel never reads them.
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

class Fold8x8x240App(IpuApp):
    """Fold 4 stride-2 channel-major streams into an 8x8x240 spatial tensor.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to the 4-stream input (4 x 240 x 512 bytes; matches
            unfold_8x8x240's raw per-channel row output).
        output_path: Optional path to write the spatial output (one row per
            channel, naive row-major [H, W] layout in the first 64 lanes).
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)

        # cr13: DST_BASE_ROW (output base). CR0 and CR1 are BOTH read-only
        # hardwired constants (0 and 1 respectively; writes are silently
        # dropped -- see CR_READ_ONLY_INITIAL_VALUES). DST_BASE_ROW is
        # nonzero (it sits after the 4-stream source region), so it cannot
        # use cr0 -- same trap fold_16x16x192/fold_32x32x144 document.
        state.regfile.set_cr(13, DST_BASE_ROW)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # cr9..cr12: per-stream SOURCE bases (s0, s1, s2, s3)
        state.regfile.set_cr(9,  SRC_BASE_ROW)
        state.regfile.set_cr(10, SRC_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, SRC_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, SRC_BASE_ROW + 3 * _STREAM_ROWS)

        # cr2/cr3: source-lane-index table for ACC.RESHAPE, packed as two
        # 4-byte halves of an LRDn pair -- [0,1,2,3] then [4,5,6,7]. Rebuilt
        # via SET at the start of every stream's 2-call block in the .asm
        # (never assumed preloaded in any LR).
        state.regfile.set_cr(2, _pack_bytes([0, 1, 2, 3]))
        state.regfile.set_cr(3, _pack_bytes([4, 5, 6, 7]))
        # cr4/cr5: stream-0 destination-lane-index table for call 0, packed
        # the same way -- [0,2,4,6] then [16,18,20,22]. Streams 1/2/3 are
        # derived from this via ADDBI (+1/+8/+9) in the .asm.
        state.regfile.set_cr(4, _pack_bytes([0, 2, 4, 6]))
        state.regfile.set_cr(5, _pack_bytes([16, 18, 20, 22]))

        # constant LRs preset here (SET requires a CR source since issue #82;
        # set_lr in the harness has no such restriction).
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
                DST_BASE, DST_ROW_BYTES, C,
            )


# -- registry declaration ---------------------------------------------------
# Declared beside the kernel so the registry needs no central list. `supports`
# is the single source of truth for this kernel's domain: it is an exact-shape
# match, since the packing geometry is baked into the .asm for this one
# (H, W, C) triple -- same convention unfold_8x8x240 and the other fold
# kernels use.


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
        f"(H, W, C) == ({H}, {W}, {C}) exactly: geometry (packing, register "
        f"layout) is fixed in the .asm for this shape."
    )


SPEC = KernelSpec(
    name="fold_8x8x240",
    op="fold",
    variant="8x8x240",
    app_class=Fold8x8x240App,
    asm="fold_8x8x240.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: unfold_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
