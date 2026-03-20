"""Generate test data for the universal pointwise convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    python -m ipu_apps.convolutions_universal.pointwise_conv_universal.gen_test_data \\
        --rows 32 --cols 32 --in-channels 16 --out-channels 16 \\
        --output-dir test_data/32x32_16to16
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes


def reference_pointwise_conv(
    input_bytes: bytes,
    kernel_bytes: bytes,
    dtype: DType,
    rows: int,
    cols: int,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """Compute the reference pointwise (1x1) convolution output.

    Input layout: interleaved by row-group and channel.
      Channel ic, row-group rg: offset = rg * in_channels * 128 + ic * 128
      Within each 128-byte chunk: multiple spatial rows packed.

    Kernel layout: kernel[oc * in_channels + ic].

    Output layout: interleaved by row-group and output channel.
      Output channel oc, row-group rg: offset = (rg * out_channels + oc) * 512
      Each output element is 4 bytes (int32 or float32 accumulator).
    """
    rows_per_chunk = 128 // cols
    row_groups = (rows * cols) // 128
    simd_width = 128  # elements per chunk

    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(row_groups * out_channels * simd_width * 4)

    for rg in range(row_groups):
        for oc in range(out_channels):
            for elem in range(simd_width):
                acc: int | float = 0
                for ic in range(in_channels):
                    ki = oc * in_channels + ic
                    # Input: rg * row_group_stride + ic * 128 + elem
                    bi = rg * in_channels * 128 + ic * 128 + elem
                    a = kernel_bytes[ki]
                    b = input_bytes[bi]
                    prod = ipu_mult(a, b, dtype)
                    acc = ipu_add(acc, prod, dtype)
                    if dtype != DType.INT8:
                        acc = float(np.float32(acc))
                out_idx = (rg * out_channels + oc) * simd_width + elem
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def generate_for_dtype(
    out_dir: Path,
    dtype: DType,
    dtype_name: str,
    rows: int,
    cols: int,
    in_channels: int,
    out_channels: int,
) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    row_groups = (rows * cols) // 128
    input_size = row_groups * in_channels * 128
    kernel_size = in_channels * out_channels

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

    # Write input
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (pad to multiple of 128 for XMEM alignment)
    kernel_padded_size = max(kernel_size, 128)
    kernel_padded_size = ((kernel_padded_size + 127) // 128) * 128
    kernel_padded = bytearray(kernel_padded_size)
    kernel_padded[: len(kernel_bytes)] = kernel_bytes
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_padded))

    # Compute and write golden output
    golden = reference_pointwise_conv(
        input_bytes, kernel_bytes, dtype, rows, cols, in_channels, out_channels
    )
    (dtype_dir / golden_name).write_bytes(golden)

    print(
        f"  {dtype_name}: input={len(input_bytes)}B, "
        f"kernel={len(kernel_bytes)}B, output={len(golden)}B"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test data for universal pointwise convolution.",
    )
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--in-channels", type=int, required=True)
    parser.add_argument("--out-channels", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: test_data/<config>)",
    )
    args = parser.parse_args()

    tag = f"{args.rows}x{args.cols}_{args.in_channels}to{args.out_channels}"
    out_dir = args.output_dir or (Path(__file__).parent / "test_data" / tag)

    print(f"Generating test data for {tag} in {out_dir}")

    generate_for_dtype(
        out_dir, DType.INT8, "int8",
        args.rows, args.cols, args.in_channels, args.out_channels,
    )
    generate_for_dtype(
        out_dir, DType.FP8_E4M3, "fp8_e4m3",
        args.rows, args.cols, args.in_channels, args.out_channels,
    )
    generate_for_dtype(
        out_dir, DType.FP8_E5M2, "fp8_e5m2",
        args.rows, args.cols, args.in_channels, args.out_channels,
    )

    print("Done.")


if __name__ == "__main__":
    main()
