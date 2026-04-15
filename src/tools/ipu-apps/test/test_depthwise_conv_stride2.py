"""Self-contained end-to-end test for stride-2 depthwise 3x3 convolution.

Generates input/kernel in memory, assembles, runs, and compares against
a Python reference that matches the IPU's arithmetic exactly.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise_conv_stride2 import (
    DepthwiseConvStride2App,
    OUTPUT_BASE_ADDR,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "depthwise_conv_stride2"
    / "depthwise_conv_stride2.asm"
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
    Output is quantised INT8: clamp(acc, -128, 127).
    Last output row (j = rows//2 - 1) is zeroed if it needs an
    out-of-bounds input row.

    Output layout: interleaved 128-byte chunks, 2 output rows per chunk,
    cols_out = cols // 2, rows_per_out_chunk = 2.
    """
    dtype = DType.INT8
    rows_per_in_chunk = 128 // cols  # =1 for cols=128
    out_rows = rows // 2
    out_cols = cols // 2
    rows_per_out_chunk = 128 // out_cols  # =2 for out_cols=64
    num_out_chunks = out_rows // rows_per_out_chunk
    group_stride_out = channels * 128

    output = bytearray(num_out_chunks * channels * 128)

    for ch in range(channels):
        for oj in range(out_rows):
            # Center input row
            center = 2 * oj + 1
            # Check if this row needs an input row beyond the image
            if center + 1 >= rows:
                # Bottom border — skip (output zero)
                val_q = 0
                # Write to output
                og = oj // rows_per_out_chunk
                olr = oj % rows_per_out_chunk
                for oc in range(out_cols):
                    out_byte = (og * channels + ch) * 128 + olr * out_cols + oc
                    output[out_byte] = 0
                continue

            for oc in range(out_cols):
                # Input column for stride-2
                ic = 2 * oc + 1  # center column (matches horizontal stride "on" = even indices from 128 elements)
                # Actually: acc.stride with h_stride=on takes elements 0,2,4,...
                # Element index in the 128-wide row = col index
                # After stride: output[oc] = conv_result[2*oc]
                # conv_result[ic_full] for ic_full = 2*oc
                ic_full = 2 * oc

                acc: int = 0
                for dr in range(3):
                    for dc in range(3):
                        ir = center + dr - 1
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


class TestDepthwiseConvStride2:

    def test_128x128x16_int8(self, tmp_path: Path) -> None:
        """128x128x16 -> 64x64x16, stride-2, INT8."""
        rows, cols, channels = 128, 128, 16

        # Generate data
        input_bytes, kernel_bytes = _gen_test_data(rows, cols, channels)
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_bytes)
        kernel_file.write_bytes(kernel_bytes)

        # Assemble
        inst_file = tmp_path / "prog.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        # Run
        app = DepthwiseConvStride2App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=50_000_000)
        assert cycles > 0

        # Read output from xmem
        num_out_chunks = rows // 4  # 32
        total_out_bytes = num_out_chunks * channels * 128
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_out_bytes)

        # Compute reference
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
                # Decode position
                chunk_idx = i // (channels * 128)
                rem = i % (channels * 128)
                ch = rem // 128
                pos = rem % 128
                out_row = chunk_idx * 2 + pos // 64
                out_col = pos % 64
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

    def test_128x128x8_int8(self, tmp_path: Path) -> None:
        """128x128x8 -> 64x64x8, stride-2, INT8 (fewer channels)."""
        rows, cols, channels = 128, 128, 8

        input_bytes, kernel_bytes = _gen_test_data(rows, cols, channels, seed=99)
        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_bytes)
        kernel_file.write_bytes(kernel_bytes)

        inst_file = tmp_path / "prog.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        app = DepthwiseConvStride2App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=50_000_000)
        assert cycles > 0

        num_out_chunks = rows // 4
        total_out_bytes = num_out_chunks * channels * 128
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_out_bytes)
        expected = reference_depthwise_conv_stride2(
            input_bytes, kernel_bytes, rows, cols, channels,
        )

        assert actual == expected, "Output mismatch for 128x128x8"
