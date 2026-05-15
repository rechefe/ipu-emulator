"""Self-contained tests for the universal depthwise 3x3 convolution (FPB=28).

Tests multiple spatial/channel configurations to verify the parameterized
assembly binary produces correct INT8 output matching ipu_math arithmetic.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "depthwise_conv_universal"
    / "depthwise_conv_universal.asm"
)


def reference_depthwise_conv_universal(
    input_bytes: bytes,
    kernel_bytes: bytes,
    rows: int,
    cols: int,
    channels: int,
) -> bytes:
    """IPU-math-accurate reference for universal depthwise 3x3 convolution.

    Input layout (interleaved by chunk):
      Channel ch, row r, col c:
        rows_per_chunk = 128 // cols
        chunk = r // rows_per_chunk
        local_row = r % rows_per_chunk
        offset = (chunk * channels + ch) * 128 + local_row * cols + c

    Kernel layout: kernel[ch * 9 + dr * 3 + dc]

    Output layout (INT8): same interleaving as input.
      byte_offset = (chunk * channels + ch) * 128 + local_row * cols + c
    """
    dtype = DType.INT8
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128
    output = bytearray(num_chunks * channels * 128)

    for ch in range(channels):
        for r in range(rows):
            for c in range(cols):
                acc: int = 0
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
                clamped = max(-128, min(127, acc))
                og = r // rows_per_chunk
                olr = r % rows_per_chunk
                out_idx = (og * channels + ch) * 128 + olr * cols + c
                output[out_idx] = clamped & 0xFF

    return bytes(output)


def _gen_test_data(
    rows: int, cols: int, channels: int, seed: int = 42,
) -> tuple[bytes, bytes]:
    """Generate random INT8 input and kernel in the correct memory layout."""
    rng = np.random.RandomState(seed)
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

    # Generate raw spatial data and pack into the chunked layout
    input_raw = rng.randint(-4, 5, size=(channels, rows, cols), dtype=np.int8)
    input_packed = bytearray(num_chunks * channels * 128)
    for ch in range(channels):
        for r in range(rows):
            for c in range(cols):
                chunk = r // rows_per_chunk
                local_row = r % rows_per_chunk
                offset = (chunk * channels + ch) * 128 + local_row * cols + c
                input_packed[offset] = np.uint8(input_raw[ch, r, c]).item()

    kernel_raw = rng.randint(-4, 5, size=channels * 9, dtype=np.int8)

    return bytes(input_packed), kernel_raw.view(np.uint8).tobytes()


class TestDepthwiseConvUniversal:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("dw_univ")
        inst_file = tmp / "depthwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "rows,cols,channels",
        [
            (16, 16, 4),    # minimal: single chunk, single partial super-block
            (16, 16, 28),   # exactly one full FPB=28 super-block
            (16, 16, 32),   # one full + one partial super-block
            (32, 32, 16),   # multi-chunk, partial super-block
            (64, 64, 56),   # two full super-blocks, larger spatial
        ],
    )
    def test_depthwise(
        self,
        inst_file: Path,
        tmp_path: Path,
        rows: int,
        cols: int,
        channels: int,
    ) -> None:
        input_packed, kernel_raw = _gen_test_data(rows, cols, channels)

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_raw)

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            dtype="INT8",
            rows=rows,
            cols=cols,
            channels=channels,
        )

        num_chunks = (rows * cols) // 128
        num_super_blocks = math.ceil(channels / 28)
        max_cyc = 2_000 * num_chunks * channels * num_super_blocks + 100_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = num_chunks * channels * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_depthwise_conv_universal(
            input_packed, kernel_raw, rows, cols, channels,
        )

        assert len(actual) == len(expected), (
            f"Output size mismatch: {len(actual)} vs {len(expected)}"
        )

        rows_per_chunk = 128 // cols
        mismatches = []
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                chunk = i // (channels * 128)
                rem = i % (channels * 128)
                ch = rem // 128
                elem = rem % 128
                local_row = elem // cols
                col = elem % cols
                row = chunk * rows_per_chunk + local_row
                mismatches.append(
                    f"  ch={ch} row={row} col={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
