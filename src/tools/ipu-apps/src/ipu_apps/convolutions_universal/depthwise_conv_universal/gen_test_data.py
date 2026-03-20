"""Generate test data for the universal depthwise convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    python -m ipu_apps.convolutions_universal.depthwise_conv_universal.gen_test_data \\
        --rows 64 --cols 64 --channels 256 \\
        --output-dir test_data/64x64x256
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes


def reference_depthwise_conv(
    input_bytes: bytes,
    kernel_bytes: bytes,
    dtype: DType,
    rows: int,
    cols: int,
    channels: int,
) -> bytes:
    """Compute the reference depthwise 3x3 convolution output.

    Input layout (interleaved by chunk group):
      Channel ch, row r, col c:
        rows_per_chunk = 128 // cols
        chunk = r // rows_per_chunk
        local_row = r % rows_per_chunk
        offset = (chunk * channels + ch) * 128 + local_row * cols + c

    Kernel layout: kernel[ch * 9 + dr * 3 + dc]

    Output layout (interleaved, 4-byte accumulators):
      Channel ch, row r, col c:
        chunk = r // rows_per_chunk
        local_row = r % rows_per_chunk
        elem = local_row * cols + c
        byte_offset = (chunk * channels + ch) * 512 + elem * 4
    """
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(num_chunks * channels * 512)

    for ch in range(channels):
        for r in range(rows):
            for c in range(cols):
                acc: int | float = 0
                for dr in range(3):
                    for dc in range(3):
                        ir = r + dr - 1
                        ic = c + dc - 1
                        if 0 <= ir < rows and 0 <= ic < cols:
                            ki = ch * 9 + dr * 3 + dc
                            a = kernel_bytes[ki]
                            ig = ir // rows_per_chunk
                            ilr = ir % rows_per_chunk
                            in_idx = (ig * channels + ch) * 128 + ilr * cols + ic
                            b = input_bytes[in_idx]
                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)
                            if dtype != DType.INT8:
                                acc = float(np.float32(acc))
                og = r // rows_per_chunk
                olr = r % rows_per_chunk
                out_elem = olr * cols + c
                out_idx = (og * channels + ch) * 512 + out_elem * 4
                struct.pack_into(fmt, output, out_idx, acc)

    return bytes(output)


def generate_for_dtype(
    out_dir: Path,
    dtype: DType,
    dtype_name: str,
    rows: int,
    cols: int,
    channels: int,
) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = rows * cols * channels
    kernel_size = channels * 9

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

    # Write input (interleaved by chunk group)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (raw contiguous — harness does padding)
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(kernel_bytes)

    # Compute and write golden output
    golden = reference_depthwise_conv(
        input_bytes, kernel_bytes, dtype, rows, cols, channels
    )
    (dtype_dir / golden_name).write_bytes(golden)

    print(
        f"  {dtype_name}: input={len(input_bytes)}B, "
        f"kernel={len(kernel_bytes)}B, output={len(golden)}B"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test data for universal depthwise convolution.",
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: test_data/<config>)",
    )
    args = parser.parse_args()

    tag = f"{args.rows}x{args.cols}x{args.channels}"
    out_dir = args.output_dir or (Path(__file__).parent / "test_data" / tag)

    print(f"Generating test data for {tag} in {out_dir}")

    generate_for_dtype(
        out_dir, DType.INT8, "int8",
        args.rows, args.cols, args.channels,
    )
    generate_for_dtype(
        out_dir, DType.FP8_E4M3, "fp8_e4m3",
        args.rows, args.cols, args.channels,
    )
    generate_for_dtype(
        out_dir, DType.FP8_E5M2, "fp8_e5m2",
        args.rows, args.cols, args.channels,
    )

    print("Done.")


if __name__ == "__main__":
    main()
