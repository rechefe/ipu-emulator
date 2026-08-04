"""Unfold 32×32×144 → 4 channel-major streams.

Rearranges a 32×32×144 spatial tensor (NHCW striped) into four 16×16
sub-grid streams (TL, TR, BL, BR), each output in channel-major FP32 format.

Usage::

    from ipu_apps.unfold_32x32x144 import Unfold32x32x144App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

H         = 32    # spatial height
W         = 32    # spatial width
C         = 144   # channels
N_STRIPES = 8     # H // (H/N_STRIPES) = 8 stripes of 4 rows each
N_STREAMS = 4     # TL, TR, BL, BR
N_OUT     = C     # output channels per stream (same as input channels)
N_TG      = 2     # token groups per channel

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

_STRIPE_ROWS = C                             # one row per channel within a stripe
_STREAM_ROWS = N_OUT * N_TG            # rows per output stream (2 token groups)
SRC_STRIDE_ROWS = 1                          # one src row per channel
DST_STRIDE_ROWS = N_TG                 # rows per output channel (one per tg)
TG1_OFF_ROWS    = 1                    # tg=1 sits one row after tg=0

SRC_BASE_ROW  = 0
ONES_BASE_ROW = SRC_BASE_ROW + N_STRIPES * _STRIPE_ROWS
DST_BASE_ROW  = ONES_BASE_ROW + 1

SRC_BASE  = SRC_BASE_ROW * ROW_BYTES
ONES_BASE = ONES_BASE_ROW * ROW_BYTES
DST_BASE  = DST_BASE_ROW * ROW_BYTES


# -- XMEM loaders -----------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write NHCW-striped input directly into XMEM at SRC_BASE.

    File layout: (8 stripes × 144 channels) rows, each 128 bytes.
    Row (stripe, ch) at offset (stripe × 144 + ch) × 128.
    Each row: 4 spatial rows × 32 columns of one channel.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState") -> None:
    """One XMEM row of FP32 1.0 for r_cyclic (the pass-through multiplier)."""
    state.xmem.write_address(ONES_BASE, bytearray(np.ones(LANES, dtype=np.float32).tobytes()))


# -- App --------------------------------------------------------------------

class Unfold32x32x144App(IpuApp):
    """Unfold 32×32×144 spatial tensor into 4 channel-major streams.

    Args:
        inst_path:  Path to assembled instruction binary.
        input_path: Path to NHCW-striped input (147,456 bytes).
        output_path: Optional path to write the 4-stream FP32 output.
        dtype:      Data type string or :class:`DType`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def setup(self, state: "IpuState") -> None:
        _load_input(state, self.input_path)
        _load_ones(state)
        # cr0..cr7: per-stripe source bases (stripe s at SRC_BASE + s × 18,432).
        # CR1 (≡1) is a read-only hardwired constant on the new architecture —
        # writes are silently dropped — so the stripe-1 base goes to CR13 (free)
        # instead. cr0=SRC_BASE+0 is 0x0 (harmless no-op). See Bug #2.
        for s in range(N_STRIPES):
            cr_idx = 13 if s == 1 else s
            state.regfile.set_cr(cr_idx, SRC_BASE_ROW + s * _STRIPE_ROWS)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE_ROW)
        # cr9..cr12: per-stream destination bases (TL, TR, BL, BR)
        state.regfile.set_cr(9,  DST_BASE_ROW)
        state.regfile.set_cr(10, DST_BASE_ROW + _STREAM_ROWS)
        state.regfile.set_cr(11, DST_BASE_ROW + 2 * _STREAM_ROWS)
        state.regfile.set_cr(12, DST_BASE_ROW + 3 * _STREAM_ROWS)
        # constant LRs preset here (SET requires CR source since issue #82)
        state.regfile.set_lr(0, 0)
        state.regfile.set_lr(1, 1)
        state.regfile.set_lr(2, 2)
        state.regfile.set_lr(3, 3)
        state.regfile.set_lr(4, 0)
        state.regfile.set_lr(5, SRC_STRIDE_ROWS)   # src stride per channel (1 row)
        state.regfile.set_lr(6, DST_STRIDE_ROWS)   # dst stride per channel (8 rows)
        state.regfile.set_lr(8, 0)
        state.regfile.set_lr(9, TG1_OFF_ROWS)      # tg=1 dst offset (4 rows)
        state.regfile.set_lr(10, 0)
        state.regfile.set_lr(11, C)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                DST_BASE, OUTPUT_ROW_BYTES, N_STREAMS * N_OUT * N_TG,
            )
