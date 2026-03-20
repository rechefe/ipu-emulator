"""Generate test data for the standard 16->16 channel convolution app (8x8).

Input layout (packed pairs, 2 channels per 128-byte chunk):
  chunk j (j=0..7): channels 2j and 2j+1
  channel 2j:   offset = j*128 + row*8 + col           (bytes 0-63)
  channel 2j+1: offset = j*128 + 64 + row*8 + col      (bytes 64-127)

Output layout (only lanes 0-63 valid, 4-byte accumulators):
  filter f, element (row*8+col):
    byte_offset = f * 256 + (row*8 + col) * 4

Kernel layout per filter f:
  Block 0 (offset f*256):     channels 0-7,  byte[ch*9 + dr*3 + dc]
  Block 1 (offset f*256+128): channels 8-15, byte[(ch-8)*9 + dr*3 + dc]
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 8
COLS = 8
IN_CHANNELS = 16
OUT_CHANNELS = 16
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9
TAPS_PER_FILTER = IN_CHANNELS * KERNEL_SIZE  # 144
CHANNELS_PER_BLOCK = 8
FILTER_BLOCK_BYTES = 128
FILTER_PADDED = 2 * FILTER_BLOCK_BYTES  # 256


def _input_offset(ch: int, row: int, col: int) -> int:
    """Compute byte offset in the packed-pair input layout."""
    chunk = ch // 2
    half = ch % 2
    return chunk * 128 + half * 64 + row * COLS + col


def _reference_conv_8x8_16to16(
    input_bytes: bytes, kernel_data: list[bytes], dtype: DType
) -> bytes:
    """Compute the reference standard convolution output (16->16, 8x8).

    Input is packed: 2 channels per 128-byte chunk.
    Output: 16 filters x 64 elements x 4-byte accumulators = 4096 bytes.
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
                                b = input_bytes[_input_offset(ch, ir, ic)]
                                prod = ipu_mult(a, b, dtype)
                                acc = ipu_add(acc, prod, dtype)
                                if dtype != DType.INT8:
                                    acc = float(np.float32(acc))
                out_elem = r * COLS + c
                out_idx = f * 64 + out_elem
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS  # 1024

    if dtype == DType.INT8:
        raw = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        raw_bytes = raw.view(np.uint8).tobytes()
        kernel_filters_raw = []
        for _ in range(OUT_CHANNELS):
            kdata = rng.randint(-128, 128, size=TAPS_PER_FILTER, dtype=np.int8)
            kernel_filters_raw.append(kdata.view(np.uint8).tobytes())
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        raw_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        raw_bytes = fp32_to_fp8_bytes(raw_fp32, dtype)
        kernel_filters_raw = []
        for _ in range(OUT_CHANNELS):
            kfp32 = rng.uniform(-1.0, 1.0, size=TAPS_PER_FILTER).astype(np.float32)
            kernel_filters_raw.append(fp32_to_fp8_bytes(kfp32, dtype))
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Pack input into paired-chunk layout
    input_packed = bytearray(NUM_CHUNKS * 128)
    for ch in range(IN_CHANNELS):
        for r in range(ROWS):
            for c_val in range(COLS):
                src_idx = ch * ROWS * COLS + r * COLS + c_val
                dst_idx = _input_offset(ch, r, c_val)
                input_packed[dst_idx] = raw_bytes[src_idx]
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(bytes(input_packed))

    # Pack kernel: 2 blocks of 128 bytes per filter
    kernel_packed = bytearray(OUT_CHANNELS * FILTER_PADDED)
    for f_idx, kf in enumerate(kernel_filters_raw):
        base = f_idx * FILTER_PADDED
        # Block 0: channels 0-7 (72 bytes)
        kernel_packed[base : base + CHANNELS_PER_BLOCK * KERNEL_SIZE] = (
            kf[: CHANNELS_PER_BLOCK * KERNEL_SIZE]
        )
        # Block 1: channels 8-15 (72 bytes)
        kernel_packed[
            base + FILTER_BLOCK_BYTES : base
            + FILTER_BLOCK_BYTES
            + CHANNELS_PER_BLOCK * KERNEL_SIZE
        ] = kf[CHANNELS_PER_BLOCK * KERNEL_SIZE :]
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_packed))

    # Compute golden output using the packed input
    golden = _reference_conv_8x8_16to16(
        bytes(input_packed), kernel_filters_raw, dtype
    )
    (dtype_dir / golden_name).write_bytes(golden)

    print(
        f"  {dtype_name}: input={len(input_packed)}B, "
        f"kernel={len(kernel_packed)}B, output={len(golden)}B"
    )


NUM_CHUNKS = IN_CHANNELS // 2  # 8


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print(f"Generating test data in {out_dir}")

    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.FP8_E4M3, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.FP8_E5M2, "fp8_e5m2")

    print("Done.")


if __name__ == "__main__":
    main()
