"""Matrix-multiplication 64×64×64 test harness.

Computes C = A × W^T where:
  A: (M=64, K=64) input   — 64 bytes per row in file; padded to 128 bytes in XMEM
  W: (N=64, K=64) weights — output-major, W[n][k] at byte n*K+k (row n = all K inputs for output n)
  C: (M=64, N=64) output  — 256 bytes per row (64 × 4-byte accumulators)

Both K=64 and N=64 are < SIMD width (128):
  - A rows zero-padded to 128 bytes in XMEM.
  - T[k] (transposed weight row) zero-padded to 128 bytes; acc[64..127] = 0.

Output packing: str_acc_reg always writes 512 bytes, but assembly uses incr lr7 256
so the valid 256 bytes (N*4) of each row are packed contiguously (FC convention).

Usage::

    from ipu_apps.matmul_64x64x64 import MatMul64x64x64App

    app = MatMul64x64x64App(
        inst_path="matmul_64x64x64.bin",
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
from ipu_emu.emulator import dump_xmem_to_binary

from ipu_apps.base import IpuApp

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState

# -- Dimensions -------------------------------------------------------------

M = 64    # rows of A / rows of C
K = 64    # cols of A / cols of W per row  (< SIMD width; A rows padded to 128B in XMEM)
N = 64    # rows of W (output neurons) / cols of C  (< SIMD width; T rows padded to 128B)

# -- Memory map -------------------------------------------------------------

INPUT_BASE_ADDR   = 0x00000   # A: M × 128 bytes (padded) = 8,192 B
WEIGHTS_BASE_ADDR = 0x20000   # T: K × 128 bytes (padded) = 8,192 B
OUTPUT_BASE_ADDR  = 0x40000   # C: M × 256 bytes (packed) = 16,384 B

OUTPUT_ROW_BYTES  = N * 4     # 256 bytes

# -- Dtype helper -----------------------------------------------------------

_DTYPE_MAP = {
    "INT8":     DType.INT8,
    "int8":     DType.INT8,
    "E4": DType.E4,
    "fp8_e4": DType.E4,
    "E5": DType.E5,
    "fp8_e5": DType.E5,
}


def parse_dtype(dtype_str: str) -> DType:
    """Parse a dtype string into a :class:`DType` enum value."""
    dt = _DTYPE_MAP.get(dtype_str)
    if dt is None:
        raise ValueError(
            f"Invalid dtype '{dtype_str}'. Supported: INT8, E4, E5"
        )
    return dt


def _load_input_padded(state: "IpuState", input_path: str | Path) -> None:
    """Load A (M×K, 64B/row) into XMEM with each row zero-padded to 128 bytes."""
    raw = Path(input_path).read_bytes()
    row_bytes = bytearray(128)
    for m in range(M):
        row_bytes[:K] = raw[m * K : m * K + K]
        row_bytes[K:] = b"\x00" * (128 - K)
        state.xmem.write_address(INPUT_BASE_ADDR + m * 128, row_bytes)


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load W[n][k] from file and write T[k] (column k of W, padded to 128B) into XMEM.

    File layout: W[n][k] at byte n*K + k  (N rows × K cols, output-major).
    XMEM layout: T[k] at address WEIGHTS_BASE_ADDR + k*128.
    T[k][n] = W[n][k]; positions N..127 stay zero (N=64 < 128).
    """
    raw = Path(weights_path).read_bytes()
    for k in range(K):
        t_row = bytearray(128)
        for n in range(N):
            t_row[n] = raw[n * K + k]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + k * 128, t_row)


class MatMul64x64x64App(IpuApp):
    """64×64×64 matrix-multiplication application harness.

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
        state.set_cr_dtype(int(self.dtype))
        # Load A with each row zero-padded to 128 bytes
        _load_input_padded(state, self.input_path)
        # Transpose W (N×K output-major) → T (K rows of 128B, N valid + zeros) in XMEM
        _load_and_transpose_weights(state, self.weights_path)
        state.regfile.set_cr(0, INPUT_BASE_ADDR)
        state.regfile.set_cr(1, WEIGHTS_BASE_ADDR)
        state.regfile.set_cr(2, OUTPUT_BASE_ADDR)

    def teardown(self, state: "IpuState") -> None:
        if self.output_path is not None:
            dump_xmem_to_binary(
                state, self.output_path,
                OUTPUT_BASE_ADDR, OUTPUT_ROW_BYTES, M,
            )
