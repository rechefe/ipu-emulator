"""End-to-end tests for stride-2 depthwise 3x3 convolution (cols <= 64).

Tests 64x64 and 32x32 inputs against a Python reference implementation.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise_conv_stride2_small import (
    DepthwiseConvStride2SmallApp,
    OUTPUT_BASE_ADDR,
)


def reference_depthwise_conv_stride2(
    input_bytes: bytes,
    kernel_bytes: bytes,
    rows: int,
    cols: int,
    channels: int,
) -> bytes:
    """Compute reference stride-2 depthwise 3x3 convolution with quantisation.

    Output row j centers at input row (2j + 1).
    Output col i takes input col (2i), i.e. h_stride "on" takes even indices.
    Output is quantised INT8: clamp(acc, -128, 127).

    Output layout matches the standard interleaved chunk format:
      out_cols = cols // 2
      rows_per_out_chunk = 128 // out_cols
      For each chunk group: [ch0(128B), ch1(128B), ...]
    """
    dtype = DType.INT8
    rows_per_in_chunk = 128 // cols
    out_rows = rows // 2
    out_cols = cols // 2
    rows_per_out_chunk = 128 // out_cols
    num_out_chunk_groups = out_rows // rows_per_out_chunk
    # Handle partial last group
    if out_rows % rows_per_out_chunk != 0:
        num_out_chunk_groups += 1

    total_out_chunks = num_out_chunk_groups * channels
    output = bytearray(total_out_chunks * 128)

    for ch in range(channels):
        for oj in range(out_rows):
            center_row = 2 * oj + 1

            for oc in range(out_cols):
                # h_stride "on" takes even indices from the 128-element conv result
                # For packed rows of width `cols`, element index = row_in_chunk * cols + col
                # Even column indices: col = 0, 2, 4, ... -> output col oc uses input col 2*oc
                ic_full = 2 * oc

                acc: int = 0
                for dr in range(3):
                    for dc in range(3):
                        ir = center_row + dr - 1
                        ic_in = ic_full + dc - 1
                        if 0 <= ir < rows and 0 <= ic_in < cols:
                            ki = ch * 9 + dr * 3 + dc
                            a = kernel_bytes[ki]

                            # Input address: interleaved chunks
                            ig = ir // rows_per_in_chunk
                            ilr = ir % rows_per_in_chunk
                            in_idx = (ig * channels + ch) * 128 + ilr * cols + ic_in
                            b = input_bytes[in_idx]

                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)

                # Quantize: clamp to int8
                val_q = max(-128, min(127, acc))

                # Output address: interleaved chunks with out_cols
                og = oj // rows_per_out_chunk
                olr = oj % rows_per_out_chunk
                out_byte = (og * channels + ch) * 128 + olr * out_cols + oc
                output[out_byte] = val_q & 0xFF

    return bytes(output)


def _gen_test_data(
    rows: int, cols: int, channels: int, seed: int = 42,
) -> tuple[bytes, bytes]:
    """Generate random INT8 input and kernel data."""
    rng = np.random.RandomState(seed)
    input_size = rows * cols * channels
    kernel_size = channels * 9
    input_data = rng.randint(-3, 4, size=input_size, dtype=np.int8)
    kernel_data = rng.randint(-3, 4, size=kernel_size, dtype=np.int8)
    return input_data.view(np.uint8).tobytes(), kernel_data.view(np.uint8).tobytes()


def _run_and_compare(
    tmp_path: Path,
    rows: int,
    cols: int,
    channels: int,
    seed: int = 42,
    max_cycles: int = 50_000_000,
) -> None:
    """Assemble, run, and compare against reference."""
    input_bytes, kernel_bytes = _gen_test_data(rows, cols, channels, seed)
    input_file = tmp_path / "input.bin"
    kernel_file = tmp_path / "kernel.bin"
    input_file.write_bytes(input_bytes)
    kernel_file.write_bytes(kernel_bytes)

    app = DepthwiseConvStride2SmallApp(
        inst_path=tmp_path / "prog.bin",
        input_path=input_file,
        kernel_path=kernel_file,
        output_path=None,
        dtype="INT8",
        rows=rows,
        cols=cols,
        channels=channels,
    )

    # Assemble with Jinja-rendered source
    asm_source = app.get_asm_source()
    inst_file = tmp_path / "prog.bin"
    assemble_to_bin_file(asm_source, str(inst_file))

    state, cycles = app.run(max_cycles=max_cycles)
    assert cycles > 0

    # Read output
    out_rows = rows // 2
    out_cols = cols // 2
    rows_per_out_chunk = 128 // out_cols
    num_out_chunk_groups = out_rows // rows_per_out_chunk
    total_out_bytes = num_out_chunk_groups * channels * 128
    actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_out_bytes)

    # Reference
    expected = reference_depthwise_conv_stride2(
        input_bytes, kernel_bytes, rows, cols, channels,
    )

    assert len(actual) == len(expected), (
        f"Output size mismatch: {len(actual)} vs {len(expected)}"
    )

    # Compare byte-by-byte with diagnostics
    mismatches = []
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            chunk_idx = i // (channels * 128)
            rem = i % (channels * 128)
            ch = rem // 128
            pos = rem % 128
            out_row = chunk_idx * rows_per_out_chunk + pos // out_cols
            out_col = pos % out_cols
            a_val = struct.unpack("b", bytes([actual[i]]))[0]
            e_val = struct.unpack("b", bytes([expected[i]]))[0]
            mismatches.append(
                f"  byte {i}: ch={ch} row={out_row} col={out_col} "
                f"got={a_val} expected={e_val}"
            )
    assert not mismatches, (
        f"{len(mismatches)} mismatches (first 20):\n"
        + "\n".join(mismatches[:20])
    )


class TestDepthwiseConvStride2Small:

    def test_64x64x16_int8(self, tmp_path: Path) -> None:
        """64x64x16 -> 32x32x16, stride-2, INT8."""
        _run_and_compare(tmp_path, rows=64, cols=64, channels=16)

    def test_64x64x8_int8(self, tmp_path: Path) -> None:
        """64x64x8 -> 32x32x8, stride-2, INT8."""
        _run_and_compare(tmp_path, rows=64, cols=64, channels=8, seed=99)

    def test_32x32x16_int8(self, tmp_path: Path) -> None:
        """32x32x16 -> 16x16x16, stride-2, INT8."""
        _run_and_compare(tmp_path, rows=32, cols=32, channels=16, seed=77)

    def test_32x32x8_int8(self, tmp_path: Path) -> None:
        """32x32x8 -> 16x16x8, stride-2, INT8."""
        _run_and_compare(tmp_path, rows=32, cols=32, channels=8, seed=55)
