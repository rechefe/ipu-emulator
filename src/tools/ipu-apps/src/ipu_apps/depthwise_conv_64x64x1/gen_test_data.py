"""Generate test data for the 64x64x1 depthwise convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Input layout: 32 chunks of 128 bytes, each containing 2 spatial rows.
  Chunk c, local-row lr, col x:
    offset = c * 128 + lr * 64 + x

Output layout: 32 chunks of 512 bytes (128 elements x 4-byte acc).
  Chunk c, element idx:
    byte_offset = c * 512 + idx * 4

Usage::

    bazel run //src/tools/ipu-apps:gen_depthwise_conv_64x64x1_test_data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 64
COLS = 64
KERNEL_H = 3
KERNEL_W = 3


def _reference_conv(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference depthwise convolution output.

    For each output pixel, accumulate
    kernel[dr][dc] * input[r+dr-1][c+dc-1] with zero-padding at borders.

    The output is stored in the same packed layout as the input
    (2 rows per 128-element chunk), with 4-byte accumulator words.

    Returns 32 * 128 * 4 = 16384 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROWS * COLS * 4)

    for r in range(ROWS):
        for c in range(COLS):
            acc: int | float = 0
            for dr in range(KERNEL_H):
                for dc in range(KERNEL_W):
                    ir = r + dr - 1
                    ic = c + dc - 1
                    if 0 <= ir < ROWS and 0 <= ic < COLS:
                        a = kernel_bytes[dr * KERNEL_W + dc]
                        # Input is packed: chunk = ir // 2, local_row = ir % 2
                        chunk = ir // 2
                        local_row = ir % 2
                        b = input_bytes[chunk * 128 + local_row * COLS + ic]
                        prod = ipu_mult(a, b, dtype)
                        acc = ipu_add(acc, prod, dtype)
            # Output is packed in same chunk layout
            out_chunk = r // 2
            out_local = r % 2
            out_elem = out_local * COLS + c
            struct.pack_into(fmt, output, (out_chunk * 128 + out_elem) * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS  # 4096
    kernel_size = KERNEL_H * KERNEL_W  # 9

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

    # Write input (packed: 32 chunks x 128 bytes)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (padded to 128 bytes for XMEM alignment)
    kernel_padded = bytearray(128)
    kernel_padded[:len(kernel_bytes)] = kernel_bytes
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_padded))

    # Compute and write golden output
    golden = _reference_conv(input_bytes, kernel_bytes, dtype)
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
