"""Generate test data for the rf_feedback app.

Tests the aaq RF feedback round-trip:
  XMEM scalar → mult.ee × 1.0 → acc.first → agg → aaq → mult.ve.aaq × data → output

Scalar values chosen to be meaningful in all 3 formats:
  INT8:     4  (byte 0x04)
  FP8_E4M3: 2.0 (byte 0x40)
  FP8_E5M2: 2.0 (byte 0x40)

Three output rows per dtype:
  Row 0: agg sum value   → aaq0 & 0xFF used as mult.ve.aaq scalar
  Row 1: agg sum inv     → aaq1 & 0xFF
  Row 2: agg sum inv_sqrt → aaq2 & 0xFF
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, fp32_to_fp8_bytes

ROW_BYTES = 128


def _ones_byte(dtype: DType) -> int:
    if dtype == DType.INT8:
        return 1
    return int.from_bytes(fp32_to_fp8_bytes(np.array([1.0], np.float32), dtype), "little")


def _agg_sum_value(acc_vals: list, dtype: DType) -> int:
    """Replicate execute_agg(sum, value): sum values, pack in dtype format, return as uint32."""
    raw = sum(acc_vals)
    if dtype == DType.INT8:
        return struct.unpack("<I", struct.pack("<i", raw))[0]
    else:
        return struct.unpack("<I", struct.pack("<f", raw))[0]


def _agg_sum_inv(acc_vals: list, dtype: DType) -> int:
    """Replicate execute_agg(sum, inv): always stores float32 bits."""
    f = float(sum(acc_vals))
    result = 1.0 / f if f != 0 else 0.0
    return struct.unpack("<I", struct.pack("<f", result))[0]


def _agg_sum_inv_sqrt(acc_vals: list, dtype: DType) -> int:
    """Replicate execute_agg(sum, inv_sqrt): always stores float32 bits."""
    f = float(sum(acc_vals))
    result = 1.0 / (f ** 0.5) if f > 0 else 0.0
    return struct.unpack("<I", struct.pack("<f", result))[0]


def compute_reference(scalar_byte: int, data_bytes: bytes, dtype: DType) -> tuple[bytes, dict]:
    """Compute expected mult.ve.aaq outputs for all three agg post-functions.

    Returns (golden_bytes, diagnostics_dict).
    golden_bytes: 3 rows × 512 bytes = 1536 bytes, FP32/INT32 little-endian.
    """
    one = _ones_byte(dtype)
    fmt = "<i" if dtype == DType.INT8 else "<f"

    # Phase 1: mult.ee(scalar, 1.0) → acc.first accumulates one lane at index 0
    # Only lane 0 has the scalar value; all other lanes are zero (SCALAR_BASE[1..127]=0)
    mult_scalar = ipu_mult(scalar_byte, one, dtype)

    # agg sum sees [mult_scalar, 0, 0, ..., 0]
    acc_vals = [mult_scalar] + [0] * (ROW_BYTES - 1)

    aaq0 = _agg_sum_value(acc_vals, dtype)
    aaq1 = _agg_sum_inv(acc_vals, dtype)
    aaq2 = _agg_sum_inv_sqrt(acc_vals, dtype)

    diag = {
        "scalar_byte": scalar_byte,
        "mult_scalar": mult_scalar,
        "aaq0": aaq0,
        "aaq1": aaq1,
        "aaq2": aaq2,
        "feedback_byte_value":    aaq0 & 0xFF,
        "feedback_byte_inv":      aaq1 & 0xFF,
        "feedback_byte_inv_sqrt": aaq2 & 0xFF,
    }

    # Phase 2: for each aaq, mult.ve.aaq reads aaq & 0xFF as scalar × each data byte
    output = bytearray(3 * ROW_BYTES * 4)

    for row_idx, aaq_raw in enumerate([aaq0, aaq1, aaq2]):
        fb_byte = aaq_raw & 0xFF
        for i, d in enumerate(data_bytes[:ROW_BYTES]):
            result = ipu_mult(fb_byte, d, dtype)
            struct.pack_into(fmt, output, (row_idx * ROW_BYTES + i) * 4, result)

    return bytes(output), diag


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    if dtype == DType.INT8:
        scalar_byte = 4
        scalar_bytes = bytearray([scalar_byte] + [0] * (ROW_BYTES - 1))
        data_arr = rng.randint(1, 128, size=ROW_BYTES, dtype=np.uint8)
        data_bytes = bytes(data_arr)
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        # 2.0 encodes as 0x40 in both FP8 formats
        scalar_byte = 0x40
        scalar_bytes = bytearray([scalar_byte] + [0] * (ROW_BYTES - 1))
        data_fp32 = rng.uniform(0.5, 1.0, size=ROW_BYTES).astype(np.float32)
        data_bytes = fp32_to_fp8_bytes(data_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    (dtype_dir / f"scalar_{dtype_name}.bin").write_bytes(scalar_bytes)
    (dtype_dir / f"data_{dtype_name}.bin").write_bytes(data_bytes)

    golden, _ = compute_reference(scalar_byte, data_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)
    print(f"  [{dtype_name}] scalar=0x{scalar_byte:02x}  data={len(data_bytes)}B  golden={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print("Generating rf_feedback test data...")
    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.E4,   "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.E5,   "fp8_e5m2")
    print("Done.")


if __name__ == "__main__":
    main()
