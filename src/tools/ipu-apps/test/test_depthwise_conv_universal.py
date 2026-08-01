"""Self-contained tests for depthwise 3x3 conv, no bias / no activation (FPB=28).

Verifies the parameterized binary produces INT8 output matching an ipu_math
reference: depthwise conv (zero-pad) + clamp, no bias, no ReLU.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_emu.ipu_state import IpuState, WideVectorArithmetic
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_BASE_ROW,
    OUTPUT_CHUNK_BYTES,
    FPB,
)


ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "ipu_apps" / "convolutions_universal"
    / "depthwise" / "depthwise_conv_universal"
    / "depthwise_conv_universal.asm"
)


def reference_depthwise(
    input_bytes: bytes,
    kernel_bytes: bytes,
    rows: int,
    cols: int,
    channels: int,
) -> bytes:
    """IPU-math-accurate reference: depthwise 3x3 conv (zero-pad), clamp only.

    Layouts (input + output) match depthwise_conv_universal_bn_activation
    (chunk-interleaved).
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
    rng = np.random.RandomState(seed)
    rows_per_chunk = 128 // cols
    num_chunks = (rows * cols) // 128

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
        tmp = tmp_path_factory.mktemp("dw")
        inst_file = tmp / "depthwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "rows,cols,channels",
        [
            (16, 16, 4),    # minimal: single chunk worth, partial super-block
            (16, 16, 28),   # exactly one full FPB=28 super-block
            (16, 16, 32),   # one full + one partial super-block
            (32, 32, 16),   # multi-chunk, partial super-block
            (64, 64, 50),   # two full super-blocks, larger spatial
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
        num_super_blocks = math.ceil(channels / FPB)
        max_cyc = 2_000 * num_chunks * channels * num_super_blocks + 100_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_bytes = num_chunks * channels * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        expected = reference_depthwise(input_packed, kernel_raw, rows, cols, channels)

        assert len(actual) == len(expected)

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
                    f"  ch={ch} row={row} col={col} got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n" + "\n".join(mismatches[:20])
        )

    def test_negative_outputs_not_clamped_to_zero(self, inst_file: Path, tmp_path: Path) -> None:
        """No ReLU: negative pre-activation sums must survive (not zeroed)."""
        rows = cols = 16
        channels = 4
        # Positive input, negative weights -> sums < 0 -> must stay negative.
        input_packed = bytes([5]) * ((rows * cols) // 128 * channels * 128)
        kernel_raw = bytes([0xFF]) * (channels * 9)   # -1 weights

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_raw)

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file, input_path=input_file, kernel_path=kernel_file,
            output_path=None, dtype="INT8",
            rows=rows, cols=cols, channels=channels,
        )
        num_chunks = (rows * cols) // 128
        state, _ = app.run(max_cycles=500_000)
        total_bytes = num_chunks * channels * OUTPUT_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)
        assert all(struct.unpack_from("b", actual, i)[0] < 0 for i in range(len(actual))), (
            "expected all-negative outputs (no ReLU applied)"
        )


class TestDepthwiseConvUniversalWideVectorDebug:
    """The SAME assembled binary must also be correct in wide-vector debug mode.

    See test_conv_universal_bn_activation.py's twin test for the rationale:
    XMEM operands are row numbers and r_cyclic operands are element indices,
    neither depending on 1 B/element (narrow) vs 4 B/element (wide) width.
    """

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("dw_wide")
        inst_file = tmp / "depthwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "rows,cols,channels",
        [
            (16, 16, 4),    # single chunk, partial super-block
            (16, 16, 32),   # one full + one partial super-block (exercises the
                             # cross-cycle lr12 super-block-row advance)
        ],
    )
    def test_same_asm_runs_in_wide_vector_debug_mode(
        self,
        inst_file: Path,
        tmp_path: Path,
        rows: int,
        cols: int,
        channels: int,
    ) -> None:
        input_packed, kernel_raw = _gen_test_data(rows, cols, channels, seed=7 + channels)

        rows_per_chunk = 128 // cols
        num_chunks = (rows * cols) // 128
        packed = bytearray(num_chunks * channels * 128 * 4)
        for i, byte in enumerate(input_packed):
            struct.pack_into("<i", packed, i * 4, struct.unpack_from("b", input_packed, i)[0])
        input_file = tmp_path / "input_wide.bin"
        input_file.write_bytes(bytes(packed))

        kernel_file = tmp_path / "kernel.bin"
        kernel_file.write_bytes(kernel_raw)

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file, input_path=input_file, kernel_path=kernel_file,
            output_path=None, dtype="INT8", rows=rows, cols=cols, channels=channels,
        )
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.INT32,
            wide_vector_quantize_output=True,
        )
        num_super_blocks = math.ceil(channels / FPB)
        max_cyc = 2_000 * num_chunks * channels * num_super_blocks + 100_000
        state, cycles = app.run(max_cycles=max_cyc, state=state)
        assert cycles > 0

        expected = reference_depthwise(input_packed, kernel_raw, rows, cols, channels)

        mismatches = []
        total = num_chunks * channels
        for i in range(total):
            actual = state.xmem.read_address((OUTPUT_BASE_ROW + i) * 512, 128)
            want = expected[i * 128:(i + 1) * 128]
            for j in range(128):
                if actual[j] != want[j]:
                    a_val = struct.unpack_from("b", actual, j)[0]
                    e_val = struct.unpack_from("b", want, j)[0]
                    mismatches.append(f"  out_row={i} elem={j} got={a_val} expected={e_val}")
        assert not mismatches, (
            f"{len(mismatches)} wide-mode mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
