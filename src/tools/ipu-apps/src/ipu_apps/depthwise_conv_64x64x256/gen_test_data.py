"""Generate test data for the 256-channel depthwise convolution app (64x64).

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

64x64 spatial, 256 channels, 3x3 depthwise (one 3x3 kernel per channel).
2 spatial rows packed per 128-byte chunk (32 row-groups).

Input layout (interleaved by group):
  group g, channel ch, local_row lr, col c:
    offset = (g * 256 + ch) * 128 + lr * 64 + c

Output layout (interleaved by group, 4-byte accumulators):
  group g, channel ch, element idx:
    byte_offset = (g * 256 + ch) * 512 + idx * 4
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 64
COLS = 64
CHANNELS = 256
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9 per channel

ROWS_PER_CHUNK = 2
NUM_GROUPS = ROWS // ROWS_PER_CHUNK  # 32


def _reference_depthwise_conv(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference depthwise convolution output.

    For each channel ch, output pixel (r, c):
      acc = sum over dr,dc: kernel[ch*9 + dr*3+dc] * input[ch][r+dr-1][c+dc-1]

    Input layout (interleaved by group, 2 rows per chunk):
      Channel ch, row r, col c:
        group = r // 2
        local_row = r % 2
        offset = (group * 256 + ch) * 128 + local_row * 64 + c

    Kernel layout:
      kernel[ch * 9 + dr * 3 + dc]

    Output layout (interleaved by group, 4-byte accumulators):
      Channel ch, row r, col c:
        group = r // 2
        local_row = r % 2
        elem = local_row * 64 + c
        byte_offset = (group * 256 + ch) * 512 + elem * 4

    Returns 32*256*512 = 4194304 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(NUM_GROUPS * CHANNELS * 512)

    for ch in range(CHANNELS):
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for dr in range(KERNEL_H):
                    for dc in range(KERNEL_W):
                        ir = r + dr - 1
                        ic = c + dc - 1
                        if 0 <= ir < ROWS and 0 <= ic < COLS:
                            ki = ch * KERNEL_SIZE + dr * KERNEL_W + dc
                            a = kernel_bytes[ki]
                            # Input address
                            ig = ir // 2
                            ilr = ir % 2
                            in_idx = (ig * CHANNELS + ch) * 128 + ilr * COLS + ic
                            b = input_bytes[in_idx]
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
                            if dtype != DType.INT8:
                                acc = float(np.float32(acc))
                # Output address
                og = r // 2
                olr = r % 2
                out_elem = olr * COLS + c
                out_idx = (og * CHANNELS + ch) * 512 + out_elem * 4
                struct.pack_into(fmt, output, out_idx, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * CHANNELS  # 1048576
    kernel_size = CHANNELS * KERNEL_SIZE  # 2304

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

    # Write input (interleaved by group)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (2304 bytes, raw contiguous -- harness does padding)
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(kernel_bytes)

    # Compute and write golden output
    golden = _reference_depthwise_conv(input_bytes, kernel_bytes, dtype)
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
