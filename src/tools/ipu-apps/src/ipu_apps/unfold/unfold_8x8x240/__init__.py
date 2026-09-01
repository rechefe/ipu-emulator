"""Unfold 8x8x240 -> 4 stride-2 streams (L5).

Rearranges an 8x8x240 spatial tensor into four stride-2 decimated streams, each
output in channel-major FP32 format. The four streams are the standard stride-2
space-to-depth decomposition: stream s takes every other spatial row and column
at phase ``(s // 2, s % 2)``. They are NOT image quadrants -- ``on``/``on_inv``
are the even/odd selector encodings of ``ACC.STRIDE``, nothing more.

Geometry is derived for L5, not copied from L3/L4:

    L3 (32x32x144): 8 stripes of 4 spatial rows x 32 cols, elements_in_row=32
    L4 (16x16x192): 2 stripes of 8 spatial rows x 16 cols, elements_in_row=16
    L5 (8x8x240):   1 stripe  of 8 spatial rows x  8 cols, elements_in_row=16

L5's spatial width is W=8, and ``elements_in_row`` only encodes 16/32/64 -- 8 is
not expressible. The whole 8x8 channel (64 elements) instead occupies the first
64 lanes of a single XMEM row, viewed by ``ACC.STRIDE`` as 4 rows x 16 cols, so
each 16-element view row holds TWO spatial rows side by side.

That pairing is what the input packing has to get right. ``ACC.STRIDE``'s
vertical selector splits view rows even/odd, so a naive row-major pack (view row
r = spatial rows 2r, 2r+1) would make ``v=on`` select spatial rows {0,1,4,5}
instead of the even rows {0,2,4,6}. The stripe therefore packs spatial rows in
the order::

    view row 0 = [row 0 | row 2]
    view row 1 = [row 1 | row 3]
    view row 2 = [row 4 | row 6]
    view row 3 = [row 5 | row 7]

i.e. ``_ROW_PACK_ORDER`` below. With that packing the four (h, v) selector pairs
yield exactly the four stride-2 phases, 16 tokens each, in row-major order.

Output layout (per-stream, one channel per row):
    Each stream writes ``N_OUT`` rows of 512 B. A stream contributes only
    ``N_TOK = 16`` valid FP32 tokens (64 B) per channel, and a row is never
    shared between channels, so each row is
    ``[16 valid FP32 tokens | 112 stale lanes]``. ``ACTIVATE.QUANTIZE identity``
    stages all 128 lanes of r_acc and ``STR_POST_AAQ_REG`` writes the full 512 B
    row, so lanes 16.. carry stale r_acc content. ``teardown`` crops each row to
    its valid prefix; ``test_unfold_8x8x240_wide.py`` pins both the crop and the
    stale-lane contract.

    This is the deliberate per-stream layout. The packed ``[k][p*n]`` variant is
    a separate deferred experiment and is NOT what this kernel emits.

Usage::

    from ipu_apps.unfold.unfold_8x8x240 import Unfold8x8x240App
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

H         = 8     # spatial height
W         = 8     # spatial width
C         = 240   # channels
N_STRIPES = 1     # a whole 8x8 channel fits in one XMEM row (64 of 128 lanes)
N_STREAMS = 4     # four stride-2 phases
N_OUT     = C     # output channels per stream (same as input channels)
N_TOK     = 16    # tokens per stream (4x4 decimated grid)

# Spatial rows, in the order they are packed into a stripe's 128-lane row.
# Slot k holds spatial row _ROW_PACK_ORDER[k] in lanes [k*W, (k+1)*W).
# Pairing (0,2) (1,3) (4,6) (5,7) is what aligns ACC.STRIDE's even/odd view-row
# split with the even/odd SPATIAL row split -- see the module docstring.
#
# This is an INPUT CONTRACT, not an implementation detail: nothing in this
# kernel (or elsewhere in the repo) produces this ordering automatically. Any
# caller staging real (non-test) input for this kernel MUST pre-permute
# spatial rows into this order first -- use :func:`pack_input_rows` rather
# than re-deriving the permutation. Getting it wrong does not look like
# garbage: it silently selects the wrong stride-2 phase per output stream (a
# plausible-looking row shuffle), which is harder to catch than an obviously
# corrupt result.
_ROW_PACK_ORDER = (0, 2, 1, 3, 4, 6, 5, 7)


def pack_input_rows(x: np.ndarray) -> np.ndarray:
    """Pack a ``[C, H, W]`` spatial tensor into ``[C, LANES]`` XMEM rows.

    Row ``ch`` holds channel ``ch``'s 8x8 grid in the first ``H*W = 64``
    lanes, with spatial rows reordered per :data:`_ROW_PACK_ORDER` so that
    ``ACC.STRIDE``'s even/odd view-row split lines up with the even/odd
    SPATIAL row split (see the module docstring). Lanes 64..127 are left
    zero. This is the one place that permutation is implemented -- callers
    should use this rather than re-deriving ``_ROW_PACK_ORDER`` themselves.
    """
    if x.shape != (C, H, W):
        raise ValueError(f"expected shape {(C, H, W)}, got {x.shape}")
    rows = np.zeros((C, LANES), dtype=np.float32)
    for slot, spatial_row in enumerate(_ROW_PACK_ORDER):
        rows[:, slot * W : (slot + 1) * W] = x[:, spatial_row, :]
    return rows

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

OUTPUT_ROW_BYTES = ROW_BYTES                 # one store = one whole row
VALID_ROW_BYTES  = N_TOK * ELEM_BYTES        # meaningful prefix of each row (64 B)

_STRIPE_ROWS = C                             # one row per channel within a stripe
_STREAM_ROWS = N_OUT                         # rows per output stream
SRC_STRIDE_ROWS = 1                          # one src row per channel
DST_STRIDE_ROWS = 1                          # one whole dst row per channel

SRC_BASE_ROW  = 0
ONES_BASE_ROW = SRC_BASE_ROW + N_STRIPES * _STRIPE_ROWS
DST_BASE_ROW  = ONES_BASE_ROW + 1

SRC_BASE  = SRC_BASE_ROW * ROW_BYTES
ONES_BASE = ONES_BASE_ROW * ROW_BYTES
DST_BASE  = DST_BASE_ROW * ROW_BYTES


# -- XMEM loaders -----------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write the packed stripe input directly into XMEM at SRC_BASE.

    File layout: C rows of LANES FP32 elements (512 B each). Row ch holds
    channel ch's 8x8 grid in the first 64 lanes, spatial rows ordered by
    ``_ROW_PACK_ORDER``; lanes 64..127 are zero padding.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState") -> None:
    """One XMEM row of FP32 1.0 for r_cyclic (the pass-through multiplier)."""
    state.xmem.write_address(
        ONES_BASE, bytearray(np.ones(LANES, dtype=np.float32).tobytes())
    )


# -- App --------------------------------------------------------------------

class Unfold8x8x240App(IpuApp):
    """Unfold an 8x8x240 spatial tensor into 4 stride-2 channel-major streams.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to the packed FP32 stripe input (C x 512 bytes).
        output_path: Optional path to write the 4-stream FP32 output.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)
        # cr0: the single stripe's source base. cr0 = SRC_BASE_ROW = 0 is a
        # harmless no-op write (CR0 is the hardwired 0).
        state.regfile.set_cr(0, SRC_BASE_ROW)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # cr9..cr12: per-stream destination bases (phases 00, 01, 10, 11)
        state.regfile.set_cr(9,  DST_BASE_ROW)
        state.regfile.set_cr(10, DST_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, DST_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, DST_BASE_ROW + 3 * _STREAM_ROWS)
        # constant LRs preset here (SET requires a CR source since issue #82)
        state.regfile.set_lr(0, 0)                 # r_cyclic slot 0 / mask / acc.stride slot 0
        state.regfile.set_lr(4, 0)                 # src row offset, += 1 per channel
        state.regfile.set_lr(5, SRC_STRIDE_ROWS)   # src stride per channel
        state.regfile.set_lr(6, DST_STRIDE_ROWS)   # dst stride per channel
        state.regfile.set_lr(8, 0)                 # dst row offset = ch
        state.regfile.set_lr(10, 0)                # channel counter
        state.regfile.set_lr(11, C)                # loop limit

    def teardown(self, state: "IpuState") -> None:
        """Dump the 4 streams, cropping each row to its N_TOK valid tokens.

        Each output row is a whole XMEM row (one channel per row, never shared).
        Only the first N_TOK lanes are valid; the rest are stale r_acc lanes that
        the store path always writes. The crop happens here so consumers get a
        dense ``[N_STREAMS, N_OUT, N_TOK]`` FP32 array.
        """
        if self.output_path is None:
            return
        raw_path = Path(self.output_path).with_suffix(".rows.bin")
        dump_xmem_to_binary(
            state, raw_path,
            DST_BASE, OUTPUT_ROW_BYTES, N_STREAMS * N_OUT,
        )
        rows = np.frombuffer(raw_path.read_bytes(), dtype=np.float32).reshape(
            N_STREAMS * N_OUT, LANES
        )
        cropped = np.ascontiguousarray(rows[:, :N_TOK])
        Path(self.output_path).write_bytes(cropped.tobytes())


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
    name="unfold_8x8x240",
    op="unfold",
    variant="8x8x240",
    app_class=Unfold8x8x240App,
    asm="unfold_8x8x240.asm",
    requires=("shape",),
    tags=("fp32-wide",),
    supports=_supports,
    build=_build,
    explain=_explain,
    caveats=lambda **params: (WIDE_VECTOR_ONLY,),
    bundle=lambda **params: unfold_query(params["shape"]).bundle,
    cost=lambda **params: 0.0,
)
