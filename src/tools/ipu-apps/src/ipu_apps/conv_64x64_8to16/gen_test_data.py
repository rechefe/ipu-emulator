"""Generate test data for the standard 8->16 channel convolution app (64x64).

Input layout: 32 groups x 8 channels x 128 bytes.
  Group g, channel ch, local_row lr, col c:
    offset = (g * 8 + ch) * 128 + lr * 64 + c

Output layout: 32 groups x 16 filters x 512 bytes (128 elements x 4-byte acc).
  Group g, filter f, element idx:
    byte_offset = ((g * 16 + f) * 128 + idx) * 4
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 64
COLS = 64
IN_CHANNELS = 8
OUT_CHANNELS = 16
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9
TAPS_PER_FILTER = IN_CHANNELS * KERNEL_SIZE  # 72
FILTER_PADDED = 128


def _reference_conv_64x64_8to16(
    input_bytes: bytes, kernel_data: list[bytes], dtype: DType
) -> bytes:
    """Compute the reference standard convolution output (8->16, 64x64).

    Input is packed: 2 spatial rows per 128-byte chunk, interleaved channels.
    Output uses the same packing with 4-byte accumulators.
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
                                # Input packed: group = ir // 2, local_row = ir % 2
                                g = ir // 2
                                lr_idx = ir % 2
                                in_idx = (g * IN_CHANNELS + ch) * 128 + lr_idx * COLS + ic
                                b = input_bytes[in_idx]
                                prod = ipu_mult(a, b, dtype)
                                acc = ipu_add(acc, prod, dtype)
                                if dtype != DType.INT8:
                                    acc = float(np.float32(acc))
                # Output packed: group = r // 2, local_row = r % 2
                out_g = r // 2
                out_lr = r % 2
                out_elem = out_lr * COLS + c
                out_idx = (out_g * OUT_CHANNELS + f) * 128 + out_elem
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS  # 32768

    if dtype == DType.INT8:
        input_data = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        input_bytes = input_data.view(np.uint8).tobytes()
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

    # Write input (packed layout, already correct from flat generation)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel: 16 filters x 128 bytes each (padded)
    kernel_packed = bytearray(OUT_CHANNELS * FILTER_PADDED)
    for f_idx, kf in enumerate(kernel_filters):
        kernel_packed[f_idx * FILTER_PADDED : f_idx * FILTER_PADDED + len(kf)] = kf
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_packed))

    # Compute and write golden output
    golden = _reference_conv_64x64_8to16(input_bytes, kernel_filters, dtype)
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
