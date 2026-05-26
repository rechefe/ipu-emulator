"""Residual add 256×144 harness.

Computes C[r] = A[r] + B[r]  for r = 0..287,
where A and B are [256 tokens, 144 channels] in interleaved channel-major layout
(288 rows × 128 bytes), and C is FP32 (288 rows × 512 bytes).

Usage::

    from ipu_apps.residual_add_256x144 import ResidualAdd256x144App
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

N_ROWS          = 288       # 256 tokens × 144 channels → 288 rows of 128 bytes
ROW_BYTES       = 128
OUTPUT_ROW_BYTES = 512

A_BASE       = 0x00000
B_BASE       = 0x10000
ONES_BASE    = 0x20000
OUTPUT_BASE  = 0x30000

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4": DType.E4,
    "fp8_e4": DType.E4,
    "E5": DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5")
    return dt


def _ones_bytes(dtype: DType) -> bytearray:
    """128 bytes each representing the value 1 in the given dtype."""
    if dtype == DType.INT8:
        return bytearray(b'\x01' * ROW_BYTES)
    ones_fp32 = np.ones(ROW_BYTES, dtype=np.float32)
    return bytearray(fp32_to_fp8_bytes(ones_fp32, dtype))


class ResidualAdd256x144App(IpuApp):
    """256×144 residual add application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_a_path = Path(self.input_a_path)
        self.input_b_path = Path(self.input_b_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))

        raw_a = Path(self.input_a_path).read_bytes()
        raw_b = Path(self.input_b_path).read_bytes()
        state.xmem.write_address(A_BASE, bytearray(raw_a))
        state.xmem.write_address(B_BASE, bytearray(raw_b))
        state.xmem.write_address(ONES_BASE, _ones_bytes(self.dtype))

        state.regfile.set_cr(0, A_BASE)
        state.regfile.set_cr(1, B_BASE)
        state.regfile.set_cr(2, ONES_BASE)
        state.regfile.set_cr(3, OUTPUT_BASE)
        state.regfile.set_cr(4, 0)
        state.regfile.set_cr(5, -128)
        state.regfile.set_cr(6, N_ROWS)
        state.regfile.set_cr(7, ROW_BYTES)
        state.regfile.set_cr(8, OUTPUT_ROW_BYTES)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_ROWS,
            )
