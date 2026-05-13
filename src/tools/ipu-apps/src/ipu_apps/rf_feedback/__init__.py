"""RF Feedback test harness.

Tests the round-trip: XMEM → mult → r_acc → agg → aaq → mult.ve.aaq → r_acc → XMEM.

Memory layout::

    SCALAR_BASE = 0x00000   128 bytes: one known nonzero value at byte 0, rest zeros
    DATA_BASE   = 0x00080   128 bytes: row of known test values
    ONES_BASE   = 0x00100   128 bytes: all dtype(1.0)
    OUTPUT_BASE = 0x00180   512*3 bytes: three output rows (value, inv, inv_sqrt)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

ROW_BYTES       = 128
OUTPUT_ROW_BYTES = 512
N_OUTPUT_ROWS   = 3   # value, inv, inv_sqrt

SCALAR_BASE = 0x00000
DATA_BASE   = 0x00080
ONES_BASE   = 0x00100
OUTPUT_BASE = 0x00180

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4":       DType.E4,
    "fp8_e4":   DType.E4,
    "E5":       DType.E5,
    "fp8_e5":   DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _ones_bytes(dtype: DType) -> bytearray:
    if dtype == DType.INT8:
        return bytearray(b'\x01' * ROW_BYTES)
    ones_fp32 = np.ones(ROW_BYTES, dtype=np.float32)
    return bytearray(fp32_to_fp8_bytes(ones_fp32, dtype))


class RfFeedbackApp(IpuApp):
    """RF feedback test application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.scalar_path = Path(self.scalar_path)
        self.data_path = Path(self.data_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        scalar_bytes = bytearray(self.scalar_path.read_bytes())
        data_bytes   = bytearray(self.data_path.read_bytes())

        state.xmem.write_address(SCALAR_BASE, scalar_bytes)
        state.xmem.write_address(DATA_BASE,   data_bytes)
        state.xmem.write_address(ONES_BASE,   _ones_bytes(self.dtype))

        state.regfile.set_cr(0,  SCALAR_BASE)
        state.regfile.set_cr(1,  DATA_BASE)
        state.regfile.set_cr(2,  ONES_BASE)
        state.regfile.set_cr(3,  OUTPUT_BASE)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUTPUT_ROWS,
            )
