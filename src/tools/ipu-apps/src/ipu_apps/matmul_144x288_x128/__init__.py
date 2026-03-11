"""Transformer matmul 144×288 harness (FFN linear 2).

Computes C[j, t] = sum_k W[j, k] * D[k, t]  for all j in [0, 144), t in [0, 256).

Usage::

    from ipu_apps.matmul_144x288_x128 import MatMul144x288x128App
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

K     = 288
N_OUT = 144
N_TG  = 2
N_TOK = 128

DATA_BASE    = 0x00000
WEIGHTS_BASE = 0x20000
OUTPUT_BASE  = 0x40000

W_STRIDE        = 384    # ceil(288/128)*128 = 3*128
OUTPUT_ROW_BYTES = 512

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


def _load_data(state: "IpuState", data_path: str | Path) -> None:
    """Load channel-major input directly into XMEM.

    File layout: K=288 channels × 256 bytes each.
    """
    raw = Path(data_path).read_bytes()
    state.xmem.write_address(DATA_BASE, bytearray(raw))


def _load_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load output-major weights into XMEM (no transposition).

    File layout: W[j][k] at byte j*K + k  (N_OUT rows × K=288 cols).
    XMEM layout per output channel j (3 chunks of 128 bytes):
      WEIGHTS_BASE + j*384        : W[j, 0..127]
      WEIGHTS_BASE + j*384 + 128  : W[j, 128..255]
      WEIGHTS_BASE + j*384 + 256  : W[j, 256..287] + 96 zero bytes
    """
    raw = Path(weights_path).read_bytes()
    for j in range(N_OUT):
        row = raw[j * K : j * K + K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE,       bytearray(row[:128]))
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 128, bytearray(row[128:256]))
        tail = bytearray(128)
        tail[:K - 256] = row[256:K]
        state.xmem.write_address(WEIGHTS_BASE + j * W_STRIDE + 256, tail)


class MatMul144x288x128App(IpuApp):
    """144×288 transformer matmul application harness."""

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.set_cr_dtype(int(self.dtype))
        _load_data(state, self.input_path)
        _load_weights(state, self.weights_path)
        state.regfile.set_cr(0, DATA_BASE)
        state.regfile.set_cr(1, WEIGHTS_BASE)
        state.regfile.set_cr(2, WEIGHTS_BASE + 128)
        state.regfile.set_cr(3, WEIGHTS_BASE + 256)
        state.regfile.set_cr(4, OUTPUT_BASE)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE, OUTPUT_ROW_BYTES, N_OUT * N_TG,
            )
