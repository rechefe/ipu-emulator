"""Generate test data for the standard 4->8 channel convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

128x128 spatial, 4 input channels, 8 output channels, 3x3 kernels.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 128
COLS = 128
IN_CHANNELS = 4
OUT_CHANNELS = 8
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9 per input channel per filter
TAPS_PER_FILTER = IN_CHANNELS * KERNEL_SIZE  # 36
FILTER_PADDED = 128  # each filter padded to 128 bytes in XMEM


def _reference_conv_4to8(
    input_bytes: bytes, kernel_data: list[bytes], dtype: DType
) -> bytes:
    """Compute the reference standard convolution output (4 -> 8 channels).

    For each output filter f, output pixel (r, c):
      acc = sum over ch,dr,dc: kernel[f][ch*9+dr*3+dc] * input[ch][r+dr-1][c+dc-1]

    Input layout (interleaved by row, 4 channels):
      row r, channel ch, col c: offset = (r * IN_CHANNELS + ch) * COLS + c

    Kernel layout per filter (36 bytes):
      kernel[f][ch * 9 + dr * 3 + dc]

    Output layout (interleaved by row, 4-byte accumulators):
      Filter f, row r, col c: offset = ((r * OUT_CHANNELS + f) * COLS + c) * 4

    Returns 128*128*8 accumulator words packed as bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROWS * COLS * OUT_CHANNELS * 4)

    for f in range(OUT_CHANNELS):
        kf = kernel_data[f]
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for ch in range(IN_CHANNELS):
                    for dr in range(KERNEL_H):
                        for dc in range(KERNEL_W):
                            ir = r + dr - 1
                            ic = c + dc - 1
                            if 0 <= ir < ROWS and 0 <= ic < COLS:
                                ki = ch * KERNEL_SIZE + dr * KERNEL_W + dc
                                a = kf[ki]
                                in_idx = (ir * IN_CHANNELS + ch) * COLS + ic
                                b = input_bytes[in_idx]
                                prod = ipu_mult(a, b, dtype)
                                acc = ipu_add(acc, prod, dtype)
                                if dtype != DType.INT8:
                                    acc = float(np.float32(acc))
                out_idx = (r * OUT_CHANNELS + f) * COLS + c
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS  # 65536

    if dtype == DType.INT8:
        input_data = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        input_bytes = input_data.view(np.uint8).tobytes()
        # Generate 8 filters, each with 36 random kernel bytes
        kernel_filters = []
        for _ in range(OUT_CHANNELS):
            kdata = rng.randint(-128, 128, size=TAPS_PER_FILTER, dtype=np.int8)
            kernel_filters.append(kdata.view(np.uint8).tobytes())
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        input_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        input_bytes = fp32_to_fp8_bytes(input_fp32, dtype)
        kernel_filters = []
        for _ in range(OUT_CHANNELS):
            kfp32 = rng.uniform(-1.0, 1.0, size=TAPS_PER_FILTER).astype(np.float32)
            kernel_filters.append(fp32_to_fp8_bytes(kfp32, dtype))
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Write input
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel: 8 filters x 128 bytes each (padded)
    kernel_packed = bytearray(OUT_CHANNELS * FILTER_PADDED)
    for f, kf in enumerate(kernel_filters):
        kernel_packed[f * FILTER_PADDED : f * FILTER_PADDED + len(kf)] = kf
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_packed))

    # Compute and write golden output
    golden = _reference_conv_4to8(input_bytes, kernel_filters, dtype)
    (dtype_dir / golden_name).write_bytes(golden)

    print(f"  {dtype_name}: input={len(input_bytes)}B, "
          f"kernel={len(kernel_packed)}B, output={len(golden)}B")


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print(f"Generating test data in {out_dir}")

    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.FP8_E4M3, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.FP8_E5M2, "fp8_e5m2")

    print("Done.")


if __name__ == "__main__":
    main()
