"""Unfold 32×32×144 → 4 channel-major streams.

Rearranges a 32×32×144 spatial tensor (NHCW striped) into four 16×16
sub-grid streams (TL, TR, BL, BR), each output in channel-major FP32 format.

Usage::

    from ipu_apps.unfold_32x32x144 import Unfold32x32x144App
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

H         = 32    # spatial height
W         = 32    # spatial width
C         = 144   # channels
N_STRIPES = 8     # H // (H/N_STRIPES) = 8 stripes of 4 rows each
N_STREAMS = 4     # TL, TR, BL, BR
N_OUT     = C     # output channels per stream (same as input channels)
N_TG      = 2     # token groups per channel

# -- Memory map -------------------------------------------------------------

SRC_BASE  = 0x00000   # NHCW striped input: 8×144×128 B = 147,456 B (ends 0x23FFF)
ONES_BASE = 0x24000   # 128 bytes of dtype-encoded 1.0 for r_cyclic (ends 0x2407F)
DST_BASE  = 0x30000   # 4 streams × 288 rows × 512 B = 589,824 B (ends 0xBBFFF)

# Per-stripe byte size: 144 ch × 128 B per ch = 18,432 B
_STRIPE_BYTES = C * 128

# Per-stream size: 288 rows × 512 B = 147,456 B
_STREAM_BYTES = N_OUT * N_TG * 512

OUTPUT_ROW_BYTES = 512   # FP32: 128 words × 4 bytes

# -- Dtype 1.0 byte encoding ------------------------------------------------

# These are the byte values representing 1.0 in each dtype.
# Multiplying any value by 1.0 is the identity operation, used here to
# pass data unchanged through the MULT stage into the accumulator.
_ONES_BYTE: dict[DType, int] = {
    DType.INT8:     0x01,   # signed int8: 1
    DType.FP8_E4M3: 0x38,   # E4M3: sign=0 exp=0111(=7, bias 7 → 2^0=1) mant=000
    DType.FP8_E5M2: 0x3C,   # E5M2: sign=0 exp=01111(=15, bias 15 → 2^0=1) mant=00
}

# -- Dtype helpers -----------------------------------------------------------

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "FP8_E4M3": DType.FP8_E4M3,
    "fp8_e4m3": DType.FP8_E4M3,
    "FP8_E5M2": DType.FP8_E5M2,
    "fp8_e5m2": DType.FP8_E5M2,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, FP8_E4M3, FP8_E5M2")
    return dt


# -- XMEM loaders -----------------------------------------------------------

def _load_input(state: "IpuState", input_path: str | Path) -> None:
    """Write NHCW-striped input directly into XMEM at SRC_BASE.

    File layout: (8 stripes × 144 channels) rows, each 128 bytes.
    Row (stripe, ch) at offset (stripe × 144 + ch) × 128.
    Each row: 4 spatial rows × 32 columns of one channel.
    """
    raw = Path(input_path).read_bytes()
    state.xmem.write_address(SRC_BASE, bytearray(raw))


def _load_ones(state: "IpuState", dtype: DType) -> None:
    """Load 128 bytes of dtype-encoded 1.0 into XMEM at ONES_BASE.

    These are loaded into r_cyclic[0..127] at startup. Every mult.ev
    then broadcasts r_cyclic[0] = 1.0 over the 128 input bytes, acting
    as an identity: the MULT result equals the input, widened to FP32/INT32.
    """
    ones = bytearray([_ONES_BYTE[dtype]] * 128)
    state.xmem.write_address(ONES_BASE, ones)


# -- App --------------------------------------------------------------------

class Unfold32x32x144App(IpuApp):
    """Unfold 32×32×144 spatial tensor into 4 channel-major streams.

    Args:
        inst_path:  Path to assembled instruction binary.
        input_path: Path to NHCW-striped input (147,456 bytes).
        output_path: Optional path to write the 4-stream FP32 output.
        dtype:      Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))
        _load_input(state, self.input_path)
        _load_ones(state, self.dtype)
        # cr0..cr7: per-stripe source bases (stripe s at SRC_BASE + s × 18,432)
        for s in range(N_STRIPES):
            state.regfile.set_cr(s, SRC_BASE + s * _STRIPE_BYTES)
        # cr8: ones base (for r_cyclic loading in assembly init)
        state.regfile.set_cr(8, ONES_BASE)
        # cr9..cr12: per-stream destination bases (TL, TR, BL, BR)
        state.regfile.set_cr(9,  DST_BASE)
        state.regfile.set_cr(10, DST_BASE + _STREAM_BYTES)
        state.regfile.set_cr(11, DST_BASE + 2 * _STREAM_BYTES)
        state.regfile.set_cr(12, DST_BASE + 3 * _STREAM_BYTES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                DST_BASE, OUTPUT_ROW_BYTES, N_STREAMS * N_OUT * N_TG,
            )
