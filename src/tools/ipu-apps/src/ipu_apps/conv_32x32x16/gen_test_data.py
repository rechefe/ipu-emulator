"""Generate test data for the 32x32 16->16 channel convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

32x32 spatial, 16 input channels, 16 output channels, 3x3 kernel.
4 spatial rows packed per 128-byte chunk (8 row-groups).

Input layout (interleaved by group):
  group g, channel ch, local_row lr, col c:
    offset = (g * 16 + ch) * 128 + lr * 32 + c

Output layout (interleaved by group, 4-byte accumulators):
  group g, filter f, element idx:
    byte_offset = (g * 16 + f) * 512 + idx * 4
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 32
COLS = 32
IN_CHANNELS = 16
OUT_CHANNELS = 16
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9
TAPS_PER_FILTER = IN_CHANNELS * KERNEL_SIZE  # 144

ROWS_PER_CHUNK = 128 // COLS  # 4
NUM_GROUPS = ROWS // ROWS_PER_CHUNK  # 8


def _reference_conv_32x32x16(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference standard convolution output (16->16, 32x32).

    For each output filter f, output pixel (r, c):
      acc = sum over ic, dr, dc:
              kernel[f*144 + ic*9 + dr*3 + dc] * input[ic][r+dr-1][c+dc-1]

    Input layout (interleaved by group, 4 rows per chunk):
      Channel ch, row r, col c:
        group = r // 4
        local_row = r % 4
        offset = (group * 16 + ch) * 128 + local_row * 32 + c

    Output layout (interleaved by group, 4-byte accumulators):
      Filter f, row r, col c:
        group = r // 4
        local_row = r % 4
        elem = local_row * 32 + c
        byte_offset = (group * 16 + f) * 512 + elem * 4

    Returns 8 * 16 * 512 = 65536 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(NUM_GROUPS * OUT_CHANNELS * 512)

    for f in range(OUT_CHANNELS):
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for ic in range(IN_CHANNELS):
                    for dr in range(KERNEL_H):
                        for dc in range(KERNEL_W):
                            ir = r + dr - 1
                            jc = c + dc - 1
                            if 0 <= ir < ROWS and 0 <= jc < COLS:
                                ki = f * TAPS_PER_FILTER + ic * KERNEL_SIZE + dr * KERNEL_W + dc
                                a = kernel_bytes[ki]
                                # Input address
                                ig = ir // ROWS_PER_CHUNK
                                ilr = ir % ROWS_PER_CHUNK
                                in_idx = (ig * IN_CHANNELS + ic) * 128 + ilr * COLS + jc
                                b = input_bytes[in_idx]
                                prod = ipu_mult(a, b, dtype)
                                acc = ipu_add(acc, prod, dtype)
                                if dtype != DType.INT8:
                                    acc = float(np.float32(acc))
                # Output address
                og = r // ROWS_PER_CHUNK
                olr = r % ROWS_PER_CHUNK
                out_elem = olr * COLS + c
                out_idx = (og * OUT_CHANNELS + f) * 512 + out_elem * 4
                struct.pack_into(fmt, output, out_idx, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS  # 16384
    kernel_size = OUT_CHANNELS * TAPS_PER_FILTER  # 2304

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

    # Write input (packed layout)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (raw contiguous -- harness does packing/padding)
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(kernel_bytes)

    # Compute and write golden output
    golden = _reference_conv_32x32x16(input_bytes, kernel_bytes, dtype)
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
