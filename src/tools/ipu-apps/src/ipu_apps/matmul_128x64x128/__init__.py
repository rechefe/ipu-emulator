"""Matrix-multiplication 128×64×128 test harness.

Computes C = A × W^T where:
  A: (M=128, K=64)  input   — 64 bytes per row in file; padded to 128 bytes in XMEM
  W: (N=128, K=64)  weights — output-major, W[n][k] at byte n*K+k (row n = all K inputs for output n)
  C: (M=128, N=128) output  — 512 bytes per row (128 × 4-byte accumulators)

K=64 < SIMD width (128): each row of A is padded to 128 bytes with zeros in XMEM so
ldr_cyclic_mult_reg (128-byte load) works unchanged. The inner loop only accesses
cyclic indices 0..63, so the padded zeros are never used in computation.

Weights are stored output-major (FC convention) and transposed during loading:
T[k] = column k of W = [W[0][k], ..., W[127][k]], written as 128-byte XMEM row k.

Usage::

    from ipu_apps.matmul_128x64x128 import MatMul128x64x128App

    app = MatMul128x64x128App(
        inst_path="matmul_128x64x128.bin",
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
K = 64    # cols of A / cols of W per row  (< SIMD width; A rows padded to 128B in XMEM)
N = 128   # rows of W (output neurons) / cols of C  (= SIMD width)

# -- Memory map -------------------------------------------------------------

INPUT_BASE_ADDR   = 0x00000   # A: M × 128 bytes (padded) = 16,384 B
WEIGHTS_BASE_ADDR = 0x20000   # T: K × 128 bytes = 64 × 128 = 8,192 B
OUTPUT_BASE_ADDR  = 0x40000   # C: M × N × 4 bytes = 65,536 B

OUTPUT_ROW_BYTES  = N * 4     # 512 bytes

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
    """Load A (M×K, 64B/row) into XMEM with each row zero-padded to 128 bytes.

    ldr_cyclic_mult_reg always loads 128 bytes. Padding ensures it reads a clean
    128-byte word for each row without crossing into the next row's data.
    """
    raw = Path(input_path).read_bytes()
    row_bytes = bytearray(128)  # reusable padded buffer
    for m in range(M):
        row_bytes[:K] = raw[m * K : m * K + K]
        row_bytes[K:] = b"\x00" * (128 - K)
        state.xmem.write_address(INPUT_BASE_ADDR + m * 128, row_bytes)


def _load_and_transpose_weights(state: "IpuState", weights_path: str | Path) -> None:
    """Load W[n][k] from file and write T[k] (column k of W) into XMEM.

    File layout: W[n][k] at byte n*K + k  (N rows × K cols, output-major).
    XMEM layout: T[k] at address WEIGHTS_BASE_ADDR + k*128, padded to 128 bytes.
    T[k][n] = W[n][k] = weight from input k to output n.
    Since N=128, no zero-padding needed.
    """
    raw = Path(weights_path).read_bytes()
    for k in range(K):
        t_row = bytearray(128)
        for n in range(N):
            t_row[n] = raw[n * K + k]
        state.xmem.write_address(WEIGHTS_BASE_ADDR + k * 128, t_row)


class MatMul128x64x128App(IpuApp):
    """128×64×128 matrix-multiplication application harness.

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
        # Transpose W (N×K output-major) → T (K rows of 128B) in XMEM
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
