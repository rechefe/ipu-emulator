"""Generate test data for the residual_add_256x144 app."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

N_ROWS   = 288
ROW_BYTES = 128

# Byte representation of 1 in each dtype (used as the r_cyclic multiplier).
_ONE_BYTE = {
    DType.INT8:     1,
    DType.E4: int.from_bytes(fp32_to_fp8_bytes(np.array([1.0], np.float32), DType.E4), 'little'),
    DType.E5: int.from_bytes(fp32_to_fp8_bytes(np.array([1.0], np.float32), DType.E5), 'little'),
}


def _reference_add(a_bytes: bytes, b_bytes: bytes, dtype: DType) -> bytes:
    """Element-wise residual add matching IPU behaviour.

    The assembly computes:
      acc.first ← A[r][i] × 1   (as int32/float via ipu_mult)
      acc       += B[r][i] × 1
    Output is FP32 (int32 stored as little-endian 4-byte words for INT8).
    """
    one = _ONE_BYTE[dtype]
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(N_ROWS * ROW_BYTES * 4)
    for r in range(N_ROWS):
        for i in range(ROW_BYTES):
            a_val = ipu_mult(a_bytes[r * ROW_BYTES + i], one, dtype)
            b_val = ipu_mult(b_bytes[r * ROW_BYTES + i], one, dtype)
            result = ipu_add(a_val, b_val, dtype)
            struct.pack_into(fmt, output, (r * ROW_BYTES + i) * 4, result)
    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    if dtype == DType.INT8:
        a_arr = rng.randint(-128, 128, size=N_ROWS * ROW_BYTES, dtype=np.int8)
        b_arr = rng.randint(-128, 128, size=N_ROWS * ROW_BYTES, dtype=np.int8)
        a_bytes = a_arr.view(np.uint8).tobytes()
        b_bytes = b_arr.view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        a_fp32 = rng.uniform(-1.0, 1.0, size=N_ROWS * ROW_BYTES).astype(np.float32)
        b_fp32 = rng.uniform(-1.0, 1.0, size=N_ROWS * ROW_BYTES).astype(np.float32)
        a_bytes = fp32_to_fp8_bytes(a_fp32, dtype)
        b_bytes = fp32_to_fp8_bytes(b_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"input_a_{dtype_name}.bin").write_bytes(a_bytes)
    (dtype_dir / f"input_b_{dtype_name}.bin").write_bytes(b_bytes)

    golden = _reference_add(a_bytes, b_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)
    print(f"  [{dtype_name}] a={len(a_bytes)}B  b={len(b_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating residual_add_256x144 test data...")
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4,   "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.E5,   "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
