"""Generate test data for the matmul_432x144_x128 app.

Creates input, weights, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    uv run python src/ipu_apps/matmul_432x144_x128/gen_test_data.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

K     = 144    # input channels
N_OUT = 432    # output channels
N_TOK = 256    # total tokens (2 groups × 128)


N_TG  = 2     # token groups
N_TPG = 128   # tokens per group


def _reference_matmul(
    data_bytes: bytes, weights_bytes: bytes, dtype: DType
) -> bytes:
    """Compute C[j][t] = sum_k W[j][k] * D[k][t] using IPU arithmetic.

    Data layout (interleaved): D[k][tg][tok] at byte (k*N_TG + tg)*N_TPG + tok.
    Weights layout:         W[j][k] at byte j*K + k  (output-major).
    Output layout (grouped): C[j][tg][tok] at word tg*N_OUT*N_TPG + j*N_TPG + tok.

    Returns N_OUT*N_TOK accumulator words packed as bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(N_OUT * N_TOK * 4)

    for j in range(N_OUT):
        for t in range(N_TOK):
            tg = t // N_TPG
            tok = t % N_TPG
            acc: int | float = 0
            for k in range(K):
                d = data_bytes[(k * N_TG + tg) * N_TPG + tok]
                w = weights_bytes[j * K + k]
                prod = ipu_mult(d, w, dtype)
                acc = ipu_add(acc, prod, dtype)
                if dtype != DType.INT8:
                    acc = float(np.float32(acc))
            struct.pack_into(fmt, output, (tg * N_OUT * N_TPG + j * N_TPG + tok) * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    if dtype == DType.INT8:
        # Generate as [K, N_TG, N_TPG], store as [N_TG, K, N_TPG] (grouped)
        data_arr = rng.randint(-128, 128, size=(K, N_TG, N_TPG), dtype=np.int8)
        data_bytes = data_arr.reshape(-1).view(np.uint8).tobytes()
        weights_arr = rng.randint(-128, 128, size=N_OUT * K, dtype=np.int8)
        weights_bytes = weights_arr.view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        data_fp32 = rng.uniform(-1.0, 1.0, size=(K, N_TG, N_TPG)).astype(np.float32)
        data_bytes = fp32_to_fp8_bytes(data_fp32.reshape(-1), dtype)
        weights_fp32 = rng.uniform(-1.0, 1.0, size=N_OUT * K).astype(np.float32)
        weights_bytes = fp32_to_fp8_bytes(weights_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(data_bytes)
    (dtype_dir / f"weights_{dtype_name}.bin").write_bytes(weights_bytes)

    golden = _reference_matmul(data_bytes, weights_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(f"  [{dtype_name}] input={len(data_bytes)}B  weights={len(weights_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating matmul_432x144_x128 test data...")
    _generate_for_dtype(out_dir, DType.INT8,     "int8")
    _generate_for_dtype(out_dir, DType.FP8_E4M3, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.FP8_E5M2, "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
