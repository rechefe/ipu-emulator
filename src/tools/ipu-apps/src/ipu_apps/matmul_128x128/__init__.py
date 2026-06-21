"""Matrix-multiplication 128×128 test harness.

Computes C = A × W^T where:
  A: (M=128, K=128) input matrix  — row-major, 128 bytes per row
  W: (N=128, K=128) weight matrix — row-major, 128 bytes per row (row n = all K inputs for output n)
  C: (M=128, N=128) output matrix — 512 bytes per row (128 × 4-byte accumulators)

Weights are stored in file as W[n][k] (output-major, matching FC convention).
Python transposes W → T before loading: T[k] = column k of W = all N output weights for input k.

Usage::

    from ipu_apps.matmul_128x128 import MatMul128x128App

    app = MatMul128x128App(
        inst_path="matmul_128x128.bin",
        input_path="input.bin",
        weights_path="weights.bin",
        output_path="output.bin",
        dtype="INT8",
    )
    state, cycles = app.run()
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ipu_emu.ipu_math import DType
from ipu_emu.emulator import load_binary_to_xmem, dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

M = 128   # rows of A / rows of C
K = 128   # cols of A / cols of W per row
N = 128   # rows of W (output neurons) / cols of C  (must equal SIMD width = 128)

# -- Memory map -------------------------------------------------------------

INPUT_BASE_ADDR   = 0x00000   # A:  M × K bytes       = 16,384 B
WEIGHTS_BASE_ADDR = 0x20000   # B:  K × N bytes       = 16,384 B
OUTPUT_BASE_ADDR  = 0x40000   # C:   M × N × 4 bytes   = 65,536 B

OUTPUT_ROW_BYTES  = N * 4     # 512 bytes (128 int32/fp32 values)

# -- Dtype helper -----------------------------------------------------------

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4": DType.E4,
    "fp8_e4": DType.E4,
    "E5": DType.E5,
    "fp8_e5": DType.E5,
}


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load W[n][k] from file and write T[k] (column k of W) into XMEM.

    File layout: W[n][k] at byte n*K + k  (N rows × K cols, output-major).
    XMEM layout: T[k] at address WEIGHTS_BASE_ADDR + k*128, padded to 128 bytes.
    T[k][n] = W[n][k] = weight from input k to output n.
    """
    raw = Path(weights_path).read_bytes()
    for k in range(K):
        t_row = bytearray(128)
        for n in range(N):
            t_row[n] = raw[n * K + k]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + k * 128, t_row)


def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a :class:`DType` enum value."""
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5"
        )
    return dt


class MatMul128x128App(IpuApp):
    """128×128 matrix-multiplication application harness.

    Args:
        inst_path:    Path to assembled instruction binary.
        input_path:   Path to input matrix A binary (M×K bytes, row-major).
        weights_path: Path to weight matrix W binary (N×K bytes, output-major W[n][k]).
        output_path:  Optional path to write output C.
        dtype:        Data type string or :class:`DType`.
    """

    def __init__(self, *, dtype: str | DType = "INT8", **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_path = Path(self.input_path)
        self.weights_path = Path(self.weights_path)
        self.dtype = parse_dtype(dtype) if isinstance(dtype, str) else dtype

    def setup(self, state: "IpuState") -> None:
        state.dtype = self.dtype
        # Load A row-major: M rows of K bytes each
        load_binary_to_xmem(state, self.input_path, INPUT_BASE_ADDR, K, M)
        # Transpose W (N×K output-major) → T (K rows of 128B) in XMEM
        _load_and_transpose_weights(state, self.weights_path)
        # CR0 (≡0) and CR1 (≡1) are read-only hardwired constants on the new
        # architecture — writes are silently dropped. INPUT_BASE_ADDR is 0x0, so
        # cr0 still reads the correct input base; the weights base is moved to
        # CR11 (a free CR) instead of CR1. See MIGRATION_CHECKLIST.md Bug #2.
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(11, WEIGHTS_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)
        state.regfile.set_cr(3, 1)
        state.regfile.set_cr(4, 128)
        state.regfile.set_cr(5, 512)
        state.regfile.set_cr(6, M * 128)
        state.regfile.set_cr(7, 0)
        state.regfile.set_cr(8, -128)
        state.regfile.set_cr(9, -1)
        state.regfile.set_cr(10, K - 1)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, M,
            )
