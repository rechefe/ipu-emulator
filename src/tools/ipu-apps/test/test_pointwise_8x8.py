"""Self-contained tests for the universal pointwise 8x8 convolution.

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
from ipu_apps.convolutions_universal.pointwise_8x8 import (
    Pointwise8x8App,
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
    / "pointwise_8x8"
    / "pointwise_8x8.asm"
)


def reference_pointwise_8x8(
    input_packed: bytes,
    kernel_raw: bytes,
    in_channels: int,
    out_channels: int,
) -> bytes:
    """Compute reference pointwise conv matching the assembly's paired processing.

    Input is in packed paired-chunk format (2 ICs per 128-byte chunk).
    Kernel is in raw format: kernel_raw[oc * in_channels + ic].

    Output: oc_pairs x 128 lanes x 4 bytes (INT32).
      Lanes 0-63: f0 (even OC), lanes 64-127: f1 (odd OC).
    """
    dtype = DType.INT8
    oc_pairs = out_channels // 2
    output = bytearray(oc_pairs * 128 * 4)

    for oc in range(out_channels):
        pair = oc // 2
        half = oc % 2  # 0 -> lanes 0-63, 1 -> lanes 64-127

        for pos in range(SPATIAL):
            acc: int = 0
            for ic in range(in_channels):
                # Read from packed input
                chunk = ic // 2
                ic_half = ic % 2
                in_idx = chunk * 128 + ic_half * 64 + pos
                b = input_packed[in_idx]

                # Read kernel weight
                a = kernel_raw[oc * in_channels + ic]

                prod = ipu_mult(a, b, dtype)
                acc = ipu_add(acc, prod, dtype)

            lane = half * 64 + pos
            out_idx = pair * 128 + lane
            struct.pack_into("<i", output, out_idx * 4, acc)

    return bytes(output)


def _gen_test_data(
    in_channels: int,
    out_channels: int,
    seed: int = 42,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Generate random test data and return (input_packed, kernel_packed, input_raw, kernel_raw)."""
    rng = np.random.RandomState(seed)

    input_raw = rng.randint(-3, 4, size=in_channels * SPATIAL, dtype=np.int8)
    input_raw_bytes = input_raw.view(np.uint8).tobytes()

    kernel_raw = rng.randint(-3, 4, size=out_channels * in_channels, dtype=np.int8)
    kernel_raw_bytes = kernel_raw.view(np.uint8).tobytes()

    input_packed = _build_input_data(input_raw_bytes, in_channels)
    kernel_packed = _build_kernel_data(kernel_raw_bytes, in_channels, out_channels)

    return input_packed, kernel_packed, input_raw_bytes, kernel_raw_bytes


class TestPointwise8x8:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        """Assemble the universal binary once for all tests."""
        tmp = tmp_path_factory.mktemp("pw8x8")
        inst_file = tmp / "pointwise_8x8.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "in_ch,out_ch",
        [
            (8, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (64, 64),
            (160, 160),
        ],
    )
    def test_pointwise(
        self, inst_file: Path, tmp_path: Path, in_ch: int, out_ch: int
    ) -> None:
        input_packed, kernel_packed, _, kernel_raw = _gen_test_data(in_ch, out_ch)

        input_file = tmp_path / "input.bin"
        kernel_file = tmp_path / "kernel.bin"
        input_file.write_bytes(input_packed)
        kernel_file.write_bytes(kernel_packed)

        app = Pointwise8x8App(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=None,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        max_cyc = 500_000 * max(in_ch, out_ch) // 8
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        oc_pairs = out_ch // 2
        total_bytes = oc_pairs * ACC_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE_ADDR, total_bytes)

        expected = reference_pointwise_8x8(
            input_packed, kernel_raw, in_ch, out_ch
        )

        assert len(actual) == len(expected), (
            f"Output size mismatch: {len(actual)} vs {len(expected)}"
        )

        mismatches = []
        for i in range(0, len(expected), 4):
            a_val = struct.unpack_from("<i", actual, i)[0]
            e_val = struct.unpack_from("<i", expected, i)[0]
            if a_val != e_val:
                lane = (i // 4) % 128
                pair = (i // 4) // 128
                oc = pair * 2 + (1 if lane >= 64 else 0)
                pos = lane % 64
                row = pos // COLS
                col = pos % COLS
                mismatches.append(
                    f"  pair={pair} oc={oc} row={row} col={col} "
                    f"got={a_val} expected={e_val}"
                )
        assert not mismatches, (
            f"{len(mismatches)} mismatches (first 20):\n"
            + "\n".join(mismatches[:20])
        )
