"""Self-contained tests for the universal residual add app.

Adds two INT8 tensors element-wise and stores results as INT32.
Tests the four MobileViT S residual connection sizes:
  64 channels (8x8x64), 96 channels (8x8x96), 128 channels (8x8x128), 160 channels (8x8x160).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_math import DType, ipu_mult, ipu_add
from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.residual_add import (
    ResidualAddApp,
    OUTPUT_BASE,
    ACC_CHUNK_BYTES,
    SPATIAL,
    _build_input_data,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "residual_add"
    / "residual_add.asm"
)


def reference_residual_add(
    a_packed: bytes,
    b_packed: bytes,
    num_channels: int,
) -> bytes:
    """Compute reference residual add matching the assembly's behavior.

    Both inputs are in paired-chunk format (2 channels per 128-byte chunk).
    Output: ch_pairs x 128 lanes x 4 bytes (INT32).
      Lanes 0-63: even channel, lanes 64-127: odd channel.

    Uses ipu_mult(1, x) + ipu_add to match exact emulator arithmetic.
    """
    dtype = DType.INT8
    one = 0x01  # INT8 representation of 1
    ch_pairs = num_channels // 2
    output = bytearray(ch_pairs * ACC_CHUNK_BYTES)

    for pair in range(ch_pairs):
        for lane in range(128):
            # lane 0-63 -> even channel, lane 64-127 -> odd channel
            in_idx = pair * 128 + lane
            a_byte = a_packed[in_idx]
            b_byte = b_packed[in_idx]

            val_a = ipu_mult(one, a_byte, dtype)
            val_b = ipu_mult(one, b_byte, dtype)
            result = ipu_add(val_a, val_b, dtype)

            out_idx = pair * 128 + lane
            struct.pack_into("<i", output, out_idx * 4, result)

    return bytes(output)


def _gen_test_data(
    num_channels: int,
    seed: int = 42,
) -> tuple[bytes, bytes]:
    """Generate random packed input data for both tensors."""
    rng = np.random.RandomState(seed)

    a_raw = rng.randint(-3, 4, size=num_channels * SPATIAL, dtype=np.int8)
    b_raw = rng.randint(-5, 6, size=num_channels * SPATIAL, dtype=np.int8)

    a_packed = _build_input_data(a_raw.view(np.uint8).tobytes(), num_channels)
    b_packed = _build_input_data(b_raw.view(np.uint8).tobytes(), num_channels)

    return a_packed, b_packed


class TestResidualAdd:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        """Assemble the universal binary once for all tests."""
        tmp = tmp_path_factory.mktemp("residual_add")
        inst_file = tmp / "residual_add.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize("num_ch", [64, 96, 128, 160])
    def test_residual_add(self, inst_file: Path, tmp_path: Path, num_ch: int) -> None:
        a_packed, b_packed = _gen_test_data(num_ch)

        a_file = tmp_path / "input_a.bin"
        b_file = tmp_path / "input_b.bin"
        a_file.write_bytes(a_packed)
        b_file.write_bytes(b_packed)

        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_channels=num_ch,
        )

        ch_pairs = num_ch // 2
        max_cyc = ch_pairs * 10 + 1000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_output = ch_pairs * ACC_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE, total_output)
        expected = reference_residual_add(a_packed, b_packed, num_ch)

        assert len(actual) == len(expected)

        mismatches = []
        for pair in range(ch_pairs):
            for lane in range(128):
                idx = pair * 128 + lane
                act_val = struct.unpack_from("<i", actual, idx * 4)[0]
                exp_val = struct.unpack_from("<i", expected, idx * 4)[0]
                if act_val != exp_val:
                    ch = pair * 2 + (0 if lane < 64 else 1)
                    pos = lane % 64
                    mismatches.append(
                        f"ch={ch} pos={pos}: got {act_val}, expected {exp_val}"
                    )

        assert not mismatches, (
            f"{len(mismatches)} mismatches for num_channels={num_ch}:\n"
            + "\n".join(mismatches[:20])
        )
