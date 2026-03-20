"""Generate test data for the pointwise 160->160 channel convolution app (8x8).

Input layout (packed pairs, 2 channels per 128-byte chunk):
  chunk j (j=0..79): channels 2j and 2j+1
  channel 2j:   offset = j*128 + row*8 + col           (bytes 0-63)
  channel 2j+1: offset = j*128 + 64 + row*8 + col      (bytes 64-127)

Output layout (paired-output, 4-byte accumulators):
  pair p: f0 in lanes 0-63, f1 in lanes 64-127
  byte_offset = p * 512 + lane * 4

Kernel layout (interleaved per IC pair, 3 blocks per OC pair):
  OC pair p (f0=2p, f1=2p+1), IC pair j (ICs 2j, 2j+1):
    block = j // 32, chunk_in_block = j % 32
    offset = p*384 + block*128 + chunk_in_block*4
    bytes: [f0[2j], f0[2j+1], f1[2j], f1[2j+1]]
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 8
COLS = 8
IN_CHANNELS = 160
OUT_CHANNELS = 160
NUM_PAIRS = OUT_CHANNELS // 2  # 80
NUM_INPUT_CHUNKS = IN_CHANNELS // 2  # 80
BLOCKS_PER_PAIR = 3
KERNEL_BLOCK_BYTES = 128
KERNEL_BYTES_PER_PAIR = BLOCKS_PER_PAIR * KERNEL_BLOCK_BYTES  # 384


def _input_offset(ch: int, row: int, col: int) -> int:
    """Compute byte offset in the packed-pair input layout."""
    chunk = ch // 2
    half = ch % 2
    return chunk * 128 + half * 64 + row * COLS + col


def _reference_pointwise_conv_8x8x160(
    input_bytes: bytes, kernel_raw: list[bytes], dtype: DType
) -> bytes:
    """Compute the reference pointwise convolution output (160->160, 8x8).

    For each output channel oc, spatial position (r, c):
      acc = sum over ic: kernel[oc][ic] * input[ic][r][c]

    Output: paired format, 80 pairs x 128 lanes x 4 bytes = 40960 bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(NUM_PAIRS * 128 * 4)

    for oc in range(OUT_CHANNELS):
        pair = oc // 2
        half = oc % 2
        kf = kernel_raw[oc]
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for ic in range(IN_CHANNELS):
                    a = kf[ic]
                    b = input_bytes[_input_offset(ic, r, c)]
                    prod = ipu_mult(a, b, dtype)
                    acc = ipu_add(acc, prod, dtype)
                    if dtype != DType.INT8:
                        acc = float(np.float32(acc))
                lane = half * 64 + r * COLS + c
                out_idx = pair * 128 + lane
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS  # 10240
    kernel_size = IN_CHANNELS  # 160 per OC

    if dtype == DType.INT8:
        raw = rng.randint(-128, 128, size=input_size, dtype=np.int8)
        raw_bytes = raw.view(np.uint8).tobytes()
        kernel_ocs_raw = []
        for _ in range(OUT_CHANNELS):
            kdata = rng.randint(-128, 128, size=kernel_size, dtype=np.int8)
            kernel_ocs_raw.append(kdata.view(np.uint8).tobytes())
        golden_name = f"out_{dtype_name}_acc_int32.bin"
    else:
        raw_fp32 = rng.uniform(-1.0, 1.0, size=input_size).astype(np.float32)
        raw_bytes = fp32_to_fp8_bytes(raw_fp32, dtype)
        kernel_ocs_raw = []
        for _ in range(OUT_CHANNELS):
            kfp32 = rng.uniform(-1.0, 1.0, size=kernel_size).astype(np.float32)
            kernel_ocs_raw.append(fp32_to_fp8_bytes(kfp32, dtype))
        golden_name = f"out_{dtype_name}_acc_fp32.bin"

    # Pack input into paired-chunk layout
    input_packed = bytearray(NUM_INPUT_CHUNKS * 128)
    for ch in range(IN_CHANNELS):
        for r in range(ROWS):
            for c_val in range(COLS):
                src_idx = ch * ROWS * COLS + r * COLS + c_val
                dst_idx = _input_offset(ch, r, c_val)
                input_packed[dst_idx] = raw_bytes[src_idx]
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(bytes(input_packed))

    # Pack kernel: interleaved per IC pair, 3 blocks per OC pair
    kernel_packed = bytearray(NUM_PAIRS * KERNEL_BYTES_PER_PAIR)
    for p in range(NUM_PAIRS):
        f0 = 2 * p
        f1 = 2 * p + 1
        for j in range(NUM_INPUT_CHUNKS):  # 80 IC pairs
            block = j // 32
            chunk_in_block = j % 32
            dst = p * KERNEL_BYTES_PER_PAIR + block * KERNEL_BLOCK_BYTES + chunk_in_block * 4
            ic_even = 2 * j
            ic_odd = 2 * j + 1
            kernel_packed[dst] = kernel_ocs_raw[f0][ic_even]
            kernel_packed[dst + 1] = kernel_ocs_raw[f0][ic_odd]
            kernel_packed[dst + 2] = kernel_ocs_raw[f1][ic_even]
            kernel_packed[dst + 3] = kernel_ocs_raw[f1][ic_odd]
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_packed))

    # Compute golden output using the packed input
    golden = _reference_pointwise_conv_8x8x160(
        bytes(input_packed), kernel_ocs_raw, dtype
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
