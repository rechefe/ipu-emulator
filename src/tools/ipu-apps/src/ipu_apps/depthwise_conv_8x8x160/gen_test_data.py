"""Generate test data for the depthwise 160-channel convolution app (8x8).

Input layout (packed pairs, 2 channels per 128-byte chunk):
  chunk j (j=0..79): channels 2j and 2j+1
  channel 2j:   offset = j*128 + row*8 + col           (bytes 0-63)
  channel 2j+1: offset = j*128 + 64 + row*8 + col      (bytes 64-127)

Output layout (paired channels, 4-byte accumulators):
  pair p: ch 2p in lanes 0-63, ch 2p+1 in lanes 64-127
  byte_offset = p * 512 + lane * 4

Kernel layout:
  Group g (offset g*128): channels g*8..(g+1)*8-1
    byte[ch_local*9 + dr*3 + dc]
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 8
COLS = 8
NUM_CHANNELS = 160
KERNEL_H = 3
KERNEL_W = 3
KERNEL_SIZE = KERNEL_H * KERNEL_W  # 9
CHANNELS_PER_GROUP = 8
NUM_KERNEL_GROUPS = NUM_CHANNELS // CHANNELS_PER_GROUP  # 20
KERNEL_GROUP_BYTES = 128
NUM_CHUNKS = NUM_CHANNELS // 2  # 80


def _input_offset(ch: int, row: int, col: int) -> int:
    """Compute byte offset in the packed-pair input layout."""
    chunk = ch // 2
    half = ch % 2
    return chunk * 128 + half * 64 + row * COLS + col


def _reference_depthwise_conv_8x8x160(
    input_bytes: bytes, kernel_data: list[bytes], dtype: DType
) -> bytes:
    """Compute the reference depthwise convolution output (160ch, 8x8).

    Each channel has its own 3x3 kernel (no cross-channel mixing).
    Output: 160 channels x 64 elements x 4-byte accumulators = 40960 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROWS * COLS * NUM_CHANNELS * 4)

    for ch in range(NUM_CHANNELS):
        kf = kernel_data[ch]
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for dr in range(KERNEL_H):
                    for dc in range(KERNEL_W):
                        ir = r + dr - 1
                        ic = c + dc - 1
                        if 0 <= ir < ROWS and 0 <= ic < COLS:
                            ki = dr * KERNEL_W + dc
                            a = kf[ki]
                            b = input_bytes[_input_offset(ch, ir, ic)]
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
                            if dtype != DType.INT8:
                                acc = float(np.float32(acc))
                out_elem = r * COLS + c
                out_idx = ch * 64 + out_elem
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * NUM_CHANNELS  # 10240

    if dtype == DType.INT8:
        raw = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        raw_bytes = raw.view(np.uint8).tobytes()
        kernel_channels_raw = []
        for _ in range(NUM_CHANNELS):
            kdata = rng.randint(-128, 128, size=KERNEL_SIZE, dtype=np.int8)
            kernel_channels_raw.append(kdata.view(np.uint8).tobytes())
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        raw_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        raw_bytes = fp32_to_fp8_bytes(raw_fp32, dtype)
        kernel_channels_raw = []
        for _ in range(NUM_CHANNELS):
            kfp32 = rng.uniform(-1.0, 1.0, size=KERNEL_SIZE).astype(np.float32)
            kernel_channels_raw.append(fp32_to_fp8_bytes(kfp32, dtype))
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Pack input into paired-chunk layout
    input_packed = bytearray(NUM_CHUNKS * 128)
    for ch in range(NUM_CHANNELS):
        for r in range(ROWS):
            for c_val in range(COLS):
                src_idx = ch * ROWS * COLS + r * COLS + c_val
                dst_idx = _input_offset(ch, r, c_val)
                input_packed[dst_idx] = raw_bytes[src_idx]
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(bytes(input_packed))

    # Pack kernel: 20 groups of 128 bytes, 8 channels per group
    kernel_packed = bytearray(NUM_KERNEL_GROUPS * KERNEL_GROUP_BYTES)
    for ch_idx, kf in enumerate(kernel_channels_raw):
        group = ch_idx // CHANNELS_PER_GROUP
        ch_local = ch_idx % CHANNELS_PER_GROUP
        dst_offset = group * KERNEL_GROUP_BYTES + ch_local * KERNEL_SIZE
        kernel_packed[dst_offset : dst_offset + KERNEL_SIZE] = kf
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_packed))

    # Compute golden output using the packed input
    golden = _reference_depthwise_conv_8x8x160(
        bytes(input_packed), kernel_channels_raw, dtype
    )
    (dtype_dir / golden_name).write_bytes(golden)

    print(
        f"  {dtype_name}: input={len(input_packed)}B, "
        f"kernel={len(kernel_packed)}B, output={len(golden)}B"
    )


def main() -> None:
    out_dir = Path(__file__).parent / "test_data_format"
    print(f"Generating test data in {out_dir}")

    _generate_for_dtype(out_dir, DType.INT8, "int8")
    _generate_for_dtype(out_dir, DType.FP8_E4M3, "fp8_e4m3")
    _generate_for_dtype(out_dir, DType.FP8_E5M2, "fp8_e5m2")

    print("Done.")


if __name__ == "__main__":
    main()
