"""Generate test data for the 2-channel convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    bazel run //src/tools/ipu-apps:gen_conv2ch_test_data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 128
COLS = 128
CHANNELS = 2
KERNEL_H = 3
KERNEL_W = 3


def _reference_conv(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference 2-channel convolution output.

    Matches the IPU assembly: for each output pixel, accumulate
    kernel[ch][dr][dc] * input[ch][r+dr-1][c+dc-1] over both channels,
    with zero-padding at borders.

    Input layout: interleaved by row (row0_ch0, row0_ch1, row1_ch0, ...).
    Kernel layout: interleaved by row
        [0..2]=row-1_ch0, [3..5]=row-1_ch1,
        [6..8]=row0_ch0,  [9..11]=row0_ch1,
        [12..14]=row+1_ch0, [15..17]=row+1_ch1.

    Returns 128*128 accumulator words (int32 or float32) packed as bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROWS * COLS * 4)

    for r in range(ROWS):
        for c in range(COLS):
            acc: int | float = 0
            for dr in range(KERNEL_H):
                for ch in range(CHANNELS):
                    for dc in range(KERNEL_W):
                        ir = r + dr - 1
                        ic = c + dc - 1
                        if 0 <= ir < ROWS and 0 <= ic < COLS:
                            # Kernel index: interleaved by row
                            # dr=0: ch0=[0,1,2], ch1=[3,4,5]
                            # dr=1: ch0=[6,7,8], ch1=[9,10,11]
                            # dr=2: ch0=[12,13,14], ch1=[15,16,17]
                            ki = dr * (CHANNELS * KERNEL_W) + ch * KERNEL_W + dc
                            a = kernel_bytes[ki]
                            # Input: row r, channel ch at offset r*256 + ch*128 + c
                            b = input_bytes[ir * COLS * CHANNELS + ch * COLS + ic]
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
            struct.pack_into(fmt, output, (r * COLS + c) * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * CHANNELS
    kernel_size = KERNEL_H * KERNEL_W * CHANNELS

    if dtype == DType.INT8:
        # Random signed int8 values stored as uint8
        input_data = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        input_bytes = input_data.view(np.uint8).tobytes()
        kernel_data = rng.randint(-128, 128, size=kernel_size, dtype=np.int8)
        kernel_bytes = kernel_data.view(np.uint8).tobytes()
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        # Random FP8 values: generate float32 in a reasonable range, convert
        input_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        input_bytes = fp32_to_fp8_bytes(input_fp32, dtype)
        kernel_fp32 = rng.uniform(-1.0, 1.0, size=kernel_size).astype(np.float32)
        kernel_bytes = fp32_to_fp8_bytes(kernel_fp32, dtype)
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Write input (interleaved by row: row0_ch0, row0_ch1, row1_ch0, ...)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (padded to 128 bytes for XMEM alignment)
    kernel_padded = bytearray(COLS)
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
