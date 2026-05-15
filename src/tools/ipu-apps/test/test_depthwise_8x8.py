"""Self-contained tests for the universal depthwise 8x8 convolution.

Tests multiple channel configurations to verify the single assembly binary
works correctly with different parameters passed via CR registers.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise_8x8 import (
    Depthwise8x8App,
    OUTPUT_BASE_ADDR,
    ACC_CHUNK_BYTES,
    _build_input_data,
    _build_kernel_data,
)

ROWS = 8
COLS = 8
SPATIAL = ROWS * COLS  # 64

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "depthwise_8x8"
    / "depthwise_8x8.asm"
)


def reference_depthwise_8x8(
    input_raw: bytes,
    kernel_raw: bytes,
    num_channels: int,
) -> bytes:
    """Compute reference depthwise 3x3 conv matching the assembly's paired processing.

    Input: input_raw[ch * 64 + row * 8 + col] per channel.
    Kernel: kernel_raw[ch * 9 + kr * 3 + kc] (row-major 3x3).

    Output: num_pairs x 128 lanes x 1 byte (INT8, clamped to [-128, 127]).
      Lanes 0-63: channel A (even), lanes 64-127: channel B (odd).
    """
    dtype = DType.INT8
    num_pairs = num_channels // 2
    output = bytearray(num_pairs * 128)

    for ch in range(num_channels):
        pair = ch // 2
        half = ch % 2  # 0 -> lanes 0-63, 1 -> lanes 64-127

        for row in range(ROWS):
            for col in range(COLS):
                acc: int = 0
                for kr in range(-1, 2):
                    for kc in range(-1, 2):
                        nr = row + kr
                        nc = col + kc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            in_idx = ch * SPATIAL + nr * COLS + nc
                            b = input_raw[in_idx]

                            tap_idx = (kr + 1) * 3 + (kc + 1)
                            a = kernel_raw[ch * 9 + tap_idx]

                            prod = ipu_mult(a, b, dtype)
                            acc = ipu_add(acc, prod, dtype)

                pos = row * COLS + col
                lane = half * 64 + pos
                out_idx = pair * 128 + lane
                clamped = max(-128, min(127, acc))
                output[out_idx] = clamped & 0xFF

    return bytes(output)


def _gen_test_data(
    num_channels: int,
    seed: int = 42,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Generate random test data and return (input_packed, kernel_packed, input_raw, kernel_raw)."""
    rng = np.random.RandomState(seed)

    input_raw = rng.randint(-3, 4, size=num_channels * SPATIAL, dtype=np.int8)
    input_raw_bytes = input_raw.view(np.uint8).tobytes()

    kernel_raw = rng.randint(-3, 4, size=num_channels * 9, dtype=np.int8)
    kernel_raw_bytes = kernel_raw.view(np.uint8).tobytes()

    input_packed = _build_input_data(input_raw_bytes, num_channels)
    kernel_packed = _build_kernel_data(kernel_raw_bytes, num_channels)

    return input_packed, kernel_packed, input_raw_bytes, kernel_raw_bytes


class TestDepthwise8x8:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        """Assemble the universal binary once for all tests."""
        tmp = tmp_path_factory.mktemp("dw8x8")
        inst_file = tmp / "depthwise_8x8.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "num_ch",
        [
            8,
            16,
            32,
            64,
            160,
        ],
    )
    def test_depthwise(
        self, inst_file: Path, tmp_path: Path, num_ch: int
    ) -> None:
        input_packed, kernel_packed, input_raw, kernel_raw = _gen_test_data(num_ch)

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_packed)

        app = Depthwise8x8App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            channels=num_ch,
        )

        max_cyc = 500_000 * max(num_ch, 8) // 8
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        num_pairs = num_ch // 2
        total_bytes = num_pairs * ACC_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)

        expected = reference_depthwise_8x8(input_raw, kernel_raw, num_ch)

        assert len(actual) == len(expected), (
            f"Output size mismatch: {len(actual)} vs {len(expected)}"
        )

        mismatches = []
        for i in range(len(expected)):
            a_val = struct.unpack_from("b", actual, i)[0]
            e_val = struct.unpack_from("b", expected, i)[0]
            if a_val != e_val:
                lane = i % 128
                pair = i // 128
                ch = pair * 2 + (1 if lane >= 64 else 0)
                pos = lane % 64
                row = pos // COLS
                col = pos % COLS
                mismatches.append(
                    f"  pair={pair} ch={ch} row={row} col={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
