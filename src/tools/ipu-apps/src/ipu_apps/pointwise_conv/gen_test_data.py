"""Generate test data for the pointwise convolution app.

Creates input, kernel, and golden-output binaries for each dtype,
matching the IPU's arithmetic exactly via ipu_math.

Usage::

    bazel run //src/tools/ipu-apps:gen_pointwise_conv_test_data
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add, fp32_to_fp8_bytes

ROWS = 128
COLS = 128
IN_CHANNELS = 8
OUT_CHANNELS = 8


def _reference_pointwise_conv(
    input_bytes: bytes, kernel_bytes: bytes, dtype: DType
) -> bytes:
    """Compute the reference pointwise (1x1) convolution output.

    For each output channel oc, output pixel (r, c):
      acc = sum over ic in [0..7]: kernel[oc*8 + ic] * input[ic][r][c]

    Input layout: interleaved by row (row0_ch0, row0_ch1, ..., row0_ch7, ...).
      Channel ic, row r, col c: offset = r * IN_CHANNELS * COLS + ic * COLS + c

    Kernel layout in r0[0..63]:
      kernel[oc * IN_CHANNELS + ic] = weight for (oc, ic)

    Output layout: interleaved by row (row0_outch0, ..., row0_outch7, ...).
      Output channel oc, row r, col c: offset = (r * OUT_CHANNELS + oc) * COLS + c

    Each output element is 4 bytes (int32 or float32 accumulator).

    Returns 128*128*8 accumulator words packed as bytes.
    """
    fmt = "<i" if dtype == DType.INT8 else "<f"
    output = bytearray(ROWS * COLS * OUT_CHANNELS * 4)

    for oc in range(OUT_CHANNELS):
        for r in range(ROWS):
            for c in range(COLS):
                acc: int | float = 0
                for ic in range(IN_CHANNELS):
                    ki = oc * IN_CHANNELS + ic
                    a = kernel_bytes[ki]
                    b = input_bytes[r * IN_CHANNELS * COLS + ic * COLS + c]
                    prod = ipu_mult(a, b, dtype)
                    acc = ipu_add(acc, prod, dtype)
                    if dtype != DType.INT8:
                        acc = float(np.float32(acc))
                out_idx = (r * OUT_CHANNELS + oc) * COLS + c
                struct.pack_into(fmt, output, out_idx * 4, acc)

    return bytes(output)


def _generate_for_dtype(out_dir: Path, dtype: DType, dtype_name: str) -> None:
    """Generate input, kernel, and golden output for one dtype."""
    rng = np.random.RandomState(42)
    dtype_dir = out_dir / dtype_name
    dtype_dir.mkdir(parents=True, exist_ok=True)

    input_size = ROWS * COLS * IN_CHANNELS
    kernel_size = IN_CHANNELS * OUT_CHANNELS

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

    # Write input (interleaved by row)
    (dtype_dir / f"input_{dtype_name}.bin").write_bytes(input_bytes)

    # Write kernel (padded to 128 bytes for XMEM alignment)
    kernel_padded = bytearray(COLS)
    kernel_padded[:len(kernel_bytes)] = kernel_bytes
    (dtype_dir / f"kernel_{dtype_name}.bin").write_bytes(bytes(kernel_padded))

    # Compute and write golden output
    golden = _reference_pointwise_conv(input_bytes, kernel_bytes, dtype)
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
