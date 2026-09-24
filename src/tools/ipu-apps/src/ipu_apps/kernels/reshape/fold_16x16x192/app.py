"""Fold 16x16x192: 4 channel-major streams -> spatial 16x16x192 (L4).

Exact inverse of :mod:`ipu_apps.kernels.reshape.unfold_16x16x192`: takes the same 4
channel-major streams (TL, TR, BL, BR) that kernel produces and reconstructs
the original NHCW-striped spatial tensor, in the same 2-stripe x 192-channel
x 128-element layout unfold's own input used.

Each stream row carries only 64 valid FP32 tokens; lanes 64..127 are
unfold's stale R_ACC padding and this kernel never reads them (see the
``.asm`` header for the exact lane layout and the ACC.RESHAPE-based inverse
mapping).

Usage::

    from ipu_apps.kernels.reshape.fold_16x16x192.app import Fold16x16x192App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernels.reshape.app import unfold_spec

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
# numbers, not byte offsets.
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

    File layout: (4 streams x 192 channels) rows of 128 FP32 elements
    (matches unfold_16x16x192's OUTPUT layout exactly). Row (stream, ch) is
    file row stream * 192 + ch. Only the first 64 elements of each row are
    meaningful; the trailing 64 elements are ignored.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState") -> None:
    """One XMEM row of FP32 1.0 for R_CYCLIC (the pass-through multiplier)."""
    state.xmem.write_address(ONES_BASE, bytearray(np.ones(LANES, dtype=np.float32).tobytes()))


def _pack_bytes(values: list[int]) -> int:
    """Pack 4 small integers (each in [0, 255]) into one little-endian 32-bit word.

    Used to build ACC.RESHAPE index tables in a CR so ``SET`` can copy them
    into an LR: the CR's 4 raw bytes become an LRDn pair's 4 index elements
    once two such CRs are SET into the two halves of the pair.
    """
    if len(values) != 4:
        raise ValueError(f"expected 4 byte values; got {len(values)}")
    return int.from_bytes(bytes(v & 0xFF for v in values), "little")


# -- App ------------------------------------------------------------------

class Fold16x16x192App(IpuApp):
    """Fold 4 channel-major streams (TL, TR, BL, BR) into a 16x16x192 spatial tensor.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to the 4-stream input (4 x 192 rows of 128 FP32 elements; matches
            unfold_16x16x192's output).
        output_path: Optional path to write the NHCW-striped spatial output
            (matches unfold_16x16x192's input).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)

        # CR14, CR13: per-stripe DESTINATION bases. CR0 and CR1 are BOTH
        # read-only hardwired constants (0 and 1 respectively; writing either
        # raises EmulatorError -- see CR_READ_ONLY_INITIAL_VALUES). unfold's own
        # CR0 usage for its stripe-0 SOURCE base only works because that
        # base happens to be 0; fold's stripe-0 DESTINATION base is
        # DST_BASE_ROW (nonzero), so CR14 is used instead of CR0.
        state.regfile.set_cr(14, DST_BASE_ROW)
        state.regfile.set_cr(13, DST_BASE_ROW + C)
        # CR8: ones base (for R_CYCLIC loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # CR9..CR12: per-stream SOURCE bases (TL, TR, BL, BR)
        state.regfile.set_cr(9,  SRC_BASE_ROW)
        state.regfile.set_cr(10, SRC_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, SRC_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, SRC_BASE_ROW + 3 * _STREAM_ROWS)

        # CR2/CR3: source-lane-index table for ACC.RESHAPE, packed as two
        # 4-byte halves of an LRDn pair -- [0,1,2,3] then [4,5,6,7]. Rebuilt
        # via SET at the start of every stream's 4-call block in the .asm
        # (never assumed preloaded in any LR).
        state.regfile.set_cr(2, _pack_bytes([0, 1, 2, 3]))
        state.regfile.set_cr(3, _pack_bytes([4, 5, 6, 7]))
        # CR4/CR5: stream-TL destination-lane-index table for call 0, packed
        # the same way -- [0,2,4,6] then [8,10,12,14]. TR/BL/BR are derived
        # from this via ADDBI (+1/+16/+17) in the .asm.
        state.regfile.set_cr(4, _pack_bytes([0, 2, 4, 6]))
        state.regfile.set_cr(5, _pack_bytes([8, 10, 12, 14]))

        # constant LRs preset here (SET requires a CR source)
        state.regfile.set_lr(0, 0)      # R_CYCLIC slot 0
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
# Declared beside the kernel so the registry needs no central list; see
# :func:`~ipu_apps.kernels.reshape.app.unfold_spec` for the
# exact-shape `supports`.

SPEC = unfold_spec(Fold16x16x192App, op="fold", h=H, w=W, c=C)
