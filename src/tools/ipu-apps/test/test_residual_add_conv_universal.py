"""Self-contained tests for the universal residual add app (wide-vector FP32).

Adds two FP32 tensors element-wise and stores results as FP32, one channel per
512-byte chunk (128 FP32 lanes). Covers the four MobileViT S residual sizes:
64, 96, 128, 160 channels.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.residual_add import (
    ResidualAddApp,
    OUTPUT_BASE,
    LANES_PER_CHUNK,
    WIDE_CHUNK_BYTES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ipu_apps"
    / "convolutions_universal"
    / "residual_add"
    / "residual_add.asm"
)


def _pack_fp32(values: np.ndarray, num_channels: int) -> bytes:
    """Pack per-channel FP32 data into 512-byte chunks (128 lanes each).

    ``values`` is shape ``(num_channels, LANES_PER_CHUNK)`` of float32.
    """
    packed = bytearray(num_channels * WIDE_CHUNK_BYTES)
    flat = values.astype("<f4").reshape(num_channels, LANES_PER_CHUNK)
    for ch in range(num_channels):
        struct.pack_into(
            f"<{LANES_PER_CHUNK}f", packed, ch * WIDE_CHUNK_BYTES, *flat[ch]
        )
    return bytes(packed)


def _gen_test_data(num_channels: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    shape = (num_channels, LANES_PER_CHUNK)
    a = rng.uniform(-4.0, 4.0, size=shape).astype(np.float32)
    b = rng.uniform(-6.0, 6.0, size=shape).astype(np.float32)
    return a, b


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
        a, b = _gen_test_data(num_ch)

        a_file = tmp_path / "input_a.bin"
        b_file = tmp_path / "input_b.bin"
        a_file.write_bytes(_pack_fp32(a, num_ch))
        b_file.write_bytes(_pack_fp32(b, num_ch))

        app = ResidualAddApp(
            inst_path=inst_file,
            input_a_path=a_file,
            input_b_path=b_file,
            output_path=None,
            num_channels=num_ch,
        )

        max_cyc = num_ch * 10 + 1000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_output = num_ch * WIDE_CHUNK_BYTES
        actual = state.xmem.read_address(OUTPUT_BASE, total_output)

        expected = (a + b).astype(np.float32)

        mismatches = []
        for ch in range(num_ch):
            for lane in range(LANES_PER_CHUNK):
                idx = ch * LANES_PER_CHUNK + lane
                act_val = struct.unpack_from("<f", actual, idx * 4)[0]
                exp_val = float(expected[ch, lane])
                if act_val != exp_val:
                    mismatches.append(
                        f"ch={ch} lane={lane}: got {act_val}, expected {exp_val}"
                    )

        assert not mismatches, (
            f"{len(mismatches)} mismatches for num_channels={num_ch}:\n"
            + "\n".join(mismatches[:20])
        )
