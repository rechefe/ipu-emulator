"""Unfold 16×16×192 → 4 channel-major streams (L4).

Rearranges a 16×16×192 spatial tensor (NHCW striped) into four 8×8
sub-grid streams (TL, TR, BL, BR), each output in channel-major FP32 format.

NOT a port of :mod:`ipu_apps.unfold_32x32x144` — the geometry differs. L3 has
8 stripes of 4 spatial rows × 32 cols; L4 has 2 stripes of 8 rows × 16 cols.
A stream therefore fills only 2 of the 4 ``ACC.STRIDE`` slots (64 tokens), so
each 512-byte output row carries **64 valid FP32 tokens (256 bytes) followed by
256 bytes of stale r_acc lanes**. ``STR_ACC_REG`` always writes the full
512-byte register (``ipu.py:446-455``), and consumers must read only the first
half of each row. This is the deliberate per-stream layout (plan §4); the
packed ``[k][p·n]`` variant is the deferred §9 experiment.

Usage::

    from ipu_apps.unfold_16x16x192 import Unfold16x16x192App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

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

SRC_BASE  = 0x00000   # NHCW striped input: 2×192×128 B = 49,152 B (ends 0x0BFFF)
ONES_BASE = 0x0C000   # 128 bytes of dtype-encoded 1.0 for r_cyclic (ends 0x0C07F)
DST_BASE  = 0x30000   # 4 streams × 192 rows × 512 B = 393,216 B (ends 0x8FFFF)

# Per-stripe byte size: 192 ch × 128 B per ch = 24,576 B
_STRIPE_BYTES = C * 128

# Per-stream size: 192 rows × 512 B = 98,304 B
_STREAM_BYTES = N_OUT * 512

OUTPUT_ROW_BYTES = 512   # FP32: 128 words × 4 bytes (only first 64 words valid)
VALID_ROW_BYTES  = N_TOK * 4   # 256 — the meaningful prefix of each output row

# XMEM .asm operands are ROW numbers (one row = 128 elements), not byte
# addresses -- see issue #179. The byte constants above stay as-is because they
# only drive the harness's direct xmem read/write calls (which bypass row
# translation); the CR/LR registers in setup() feed the .asm's XMEM
# instructions instead, so they carry addresses and strides converted to rows.
ROW_SIZE_BYTES = 128
SRC_BASE_ROW    = SRC_BASE // ROW_SIZE_BYTES
ONES_BASE_ROW   = ONES_BASE // ROW_SIZE_BYTES
DST_BASE_ROW    = DST_BASE // ROW_SIZE_BYTES
_STRIPE_ROWS    = _STRIPE_BYTES // ROW_SIZE_BYTES
_STREAM_ROWS    = _STREAM_BYTES // ROW_SIZE_BYTES
SRC_STRIDE_ROWS = 1                                      # 128 B per channel = 1 row
DST_STRIDE_ROWS = 512 // ROW_SIZE_BYTES                  # 4: dst stride per channel

# -- Dtype 1.0 byte encoding ------------------------------------------------

# Byte values representing 1.0 in each dtype. Multiplying by 1.0 is the identity,
# used here to pass data unchanged through the MULT stage into the accumulator.
_ONES_BYTE: dict[DType, int] = {
    DType.INT8: 0x01,   # signed int8: 1
    DType.E4:   0x38,   # E4M3: sign=0 exp=0111(=7, bias 7 → 2^0=1) mant=000
    DType.E5:   0x3C,   # E5M2: sign=0 exp=01111(=15, bias 15 → 2^0=1) mant=00
}

_DTYPE_MAP = {
    "INT8":   DType.INT8,
    "int8":   DType.INT8,
    "E4":     DType.E4,
    "fp8_e4": DType.E4,
    "E5":     DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


# -- XMEM loaders -----------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write NHCW-striped input directly into XMEM at SRC_BASE.

    File layout: (2 stripes × 192 channels) rows, each 128 bytes.
    Row (stripe, ch) at offset (stripe × 192 + ch) × 128.
    Each row: 8 spatial rows × 16 columns of one channel.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState", dtype: DType) -> None:
    """Load 128 bytes of dtype-encoded 1.0 into XMEM at ONES_BASE.

    Loaded into r_cyclic[0..127] at startup; MULT.RC.VV then multiplies each
    input byte by 1.0, acting as an identity that widens to FP32/INT32.
    """
    state.xmem.write_address(ONES_BASE, bytearray([_ONES_BYTE[dtype]] * 128))


# -- App --------------------------------------------------------------------

class Unfold16x16x192App(IpuApp):
    """Unfold a 16×16×192 spatial tensor into 4 channel-major streams.

    Args:
        inst_path:   Path to assembled instruction binary.
        input_path:  Path to NHCW-striped input (49,152 bytes).
        output_path: Optional path to write the 4-stream FP32 output.
        dtype:       Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        _load_input(state, self.input_path)
        _load_ones(state, self.dtype)
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
