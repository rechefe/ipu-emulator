"""Generate test data for the 128x128x64→16 pointwise convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Input layout: interleaved by row-group (1 spatial row per 128-byte chunk).
  Row-group rg, channel ic, col c:
    offset = rg * ROW_GROUP_STRIDE + ic * 128 + c

Kernel layout: kernel[oc * IN_CHANNELS + ic] for oc=0..15, ic=0..63

Output layout: interleaved by row-group.
  Row-group rg, output channel oc, col c:
    byte_offset = (rg * OUT_CHANNELS + oc) * 128 * 4 + c * 4

Usage::

    bazel run //src/tools/ipu-apps:gen_pointwise_conv_128x128_64to16_test_data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 128
COLS = 128
IN_CHANNELS = 64
OUT_CHANNELS = 16

ROWS_PER_CHUNK = 128 // COLS  # 1
ROW_GROUPS = ROWS // ROWS_PER_CHUNK  # 128
ROW_GROUP_STRIDE = IN_CHANNELS * 128  # 8192
SIMD_WIDTH = 128  # elements per SIMD operation


def _reference_pointwise_conv(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference pointwise (1x1) convolution output.

    For each output channel oc, spatial position (r, c):
      acc = sum over ic in [0..63]: kernel[oc*64 + ic] * input[ic][r][c]

    Input layout: row-group interleaved, 1 row per 128-byte chunk.
      Row-group rg, channel ic, element idx (col c):
        offset = rg * 8192 + ic * 128 + idx

    Output layout: row-group interleaved, 128 elements per store (1 row x 128 cols).
      Row-group rg, output channel oc, element idx:
        byte_offset = (rg * 16 + oc) * 128 * 4 + idx * 4

    Returns 128*16*128 accumulator words packed as bytes = 1048576 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROW_GROUPS * OUT_CHANNELS * SIMD_WIDTH * 4)

    for rg in range(ROW_GROUPS):
        for oc in range(OUT_CHANNELS):
            for elem in range(SIMD_WIDTH):  # 128 elements = 1 row x 128 cols
                acc: int | float = 0
                for ic in range(IN_CHANNELS):
                    ki = oc * IN_CHANNELS + ic
                    a = kernel_bytes[ki]
                    b = input_bytes[rg * ROW_GROUP_STRIDE + ic * 128 + elem]
                    prod = ipu_mult(a, b, dtype)
                    acc = ipu_add(acc, prod, dtype)
                    if dtype != DType.INT8:
                        acc = float(np.float32(acc))
                out_offset = (rg * OUT_CHANNELS + oc) * SIMD_WIDTH + elem
                struct.pack_into(fmt, output, out_offset * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROW_GROUPS * IN_CHANNELS * 128  # 1048576
    kernel_size = IN_CHANNELS * OUT_CHANNELS  # 1024

    if dtype == DType.INT8:
        input_data = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        input_bytes = input_data.view(np.uint8).tobytes()
        kernel_data = rng.randint(-128, 128, size=kernel_size, dtype=np.int8)
        kernel_bytes = kernel_data.view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        input_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        input_bytes = fp32_to_fp8_bytes(input_fp32, dtype)
        kernel_fp32 = rng.uniform(-1.0, 1.0, size=kernel_size).astype(np.float32)
        kernel_bytes = fp32_to_fp8_bytes(kernel_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Write input (interleaved by row-group)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (1024 bytes)
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(kernel_bytes)

    # Compute and write golden output
    golden = _reference_pointwise_conv(input_bytes, kernel_bytes, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(f"  {dtype_name}: input={len(input_bytes)}B, "
          f"kernel={len(kernel_bytes)}B, output={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print(f"Generating test data in {out_dir}")

    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.FP8_E4M3, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.FP8_E5M2, "fp8_e5m2")

    print("Done.")


if __name__ == "__main__":
    main()
