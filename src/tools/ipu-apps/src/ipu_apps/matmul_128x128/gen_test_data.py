"""Generate test data for the matmul_128x128 app.

Creates input, weights, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    bazel run //src/tools/ipu-apps:gen_matmul_128x128_test_data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

M = 128   # rows of A
K = 128   # cols of A / rows of B
N = 128   # cols of B


def _reference_matmul(
    input_bytes: bytes, weights_bytes: bytes, dtype: DType
) -> bytes:
    """Compute reference C = A × W^T using IPU arithmetic.

    Input layout (A):   row-major, A[m][k] at byte m*K + k.
    Weights layout (W): output-major, W[n][k] at byte n*K + k.
    Output layout (C):  row-major, C[m][n] at word m*N + n (4 bytes each).

    Returns M*N accumulator words packed as bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(M * N * 4)

    for m in range(M):
        for n in range(N):
            acc: int | float = 0
            for k in range(K):
                a = input_bytes[m * K + k]
                w = weights_bytes[n * K + k]
                prod = ipu_mult(a, w, dtype)
                acc = ipu_add(acc, prod, dtype)
                if dtype != DType.INT8:
                    acc = float(np.float32(acc))
            struct.pack_into(fmt, output, (m * N + n) * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, weights, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    if dtype == DType.INT8:
        input_data = rng.randint(-128, 128, size=M * K, dtype=np.int8)
        input_bytes = input_data.view(np.uint8).tobytes()
        # W[n][k]: shape (N, K) — row n = all K inputs for output n
        weights_data = rng.randint(-128, 128, size=N * K, dtype=np.int8)
        weights_bytes = weights_data.view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        input_fp32 = rng.uniform(-1.0, 1.0, size=M * K).astype(np.float32)
        input_bytes = fp32_to_fp8_bytes(input_fp32, dtype)
        # W[n][k]: shape (N, K) — row n = all K inputs for output n
        weights_fp32 = rng.uniform(-1.0, 1.0, size=N * K).astype(np.float32)
        weights_bytes = fp32_to_fp8_bytes(weights_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)
    (dtype_dir / f"weights_{dtype_name}.bin").write_bytes(weights_bytes)

    golden = _reference_matmul(input_bytes, weights_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(f"  [{dtype_name}] input={len(input_bytes)}B  weights={len(weights_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating matmul_128x128 test data...")
    _generate_for_dtype(out_dir, DType.INT8,     "int8")
    _generate_for_dtype(out_dir, DType.E4, "fp8_e4")
    _generate_for_dtype(out_dir, DType.E5, "fp8_e5")
    print("Done.")


if __name__ == "__main__":
    main()
