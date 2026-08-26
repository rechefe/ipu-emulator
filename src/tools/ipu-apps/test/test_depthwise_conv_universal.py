"""Self-contained tests for depthwise 3x3 conv, no bias / no activation (FP32).

Runtime-generates random FP32 weights and inputs, runs the emulator, and
compares against a real ``torch.nn.functional.conv2d`` (groups=channels)
reference (tolerance-based, since IPU FP32 accumulation order differs from
PyTorch's).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    FPB,
)
from ipu_apps.convolutions_universal import unpack_output_chunked

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "ipu_apps" / "convolutions_universal"
    / "depthwise" / "depthwise_conv_universal"
    / "depthwise_conv_universal.asm"
)

_TOL = 1e-2


def reference_depthwise(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: depthwise 3x3 conv (groups=channels), no bias/ReLU."""
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    return F.conv2d(x, w, padding=1, groups=channels).squeeze(0).numpy()


class TestDepthwiseConvUniversal:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("dw")
        inst_file = tmp / "depthwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "channels,height,width",
        [
            (4, 16, 16),    # minimal: single chunk worth, partial super-block
            (28, 16, 16),   # exactly one full FPB=28 super-block
            (32, 16, 16),   # one full + one partial super-block
            (16, 32, 32),   # multi-chunk, partial super-block
            (50, 64, 64),   # two full super-blocks, larger spatial
            (4, 8, 128),    # cols=128: one packed row per chunk (Partition.P0)
            (30, 16, 128),  # cols=128, multi-chunk + super-block spanning
            # Non-power-of-2 / padding-heavy shapes.
            (4, 8, 8),
            (4, 5, 5),
        ],
    )
    def test_depthwise(
        self,
        inst_file: Path,
        tmp_path: Path,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        rng = np.random.RandomState(42 + channels * 7 + height + width)
        weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(channels, height, width) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            height=height, width=width, channels=channels,
        )

        num_super_blocks = math.ceil(channels / FPB)
        max_cyc = 2_000 * app.num_chunks * channels * num_super_blocks + 100_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_elements = app.num_chunks * channels * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        padded_out = unpack_output_chunked(raw, channels, app.rows, app.cols)
        actual = padded_out[:, :height, :width]
        expected = reference_depthwise(weights, input_chw)

        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for channels={channels} {height}x{width}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_negative_outputs_not_clamped_to_zero(self, inst_file: Path, tmp_path: Path) -> None:
        """No ReLU: negative pre-activation sums must survive (not zeroed)."""
        height = width = 16
        channels = 4
        weights = -np.ones((channels, 3, 3), dtype=np.float32)
        input_chw = np.full((channels, height, width), 5.0, dtype=np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file, input_path=input_file, kernel=weights,
            output_path=None, height=height, width=width, channels=channels,
        )
        state, _ = app.run(max_cycles=500_000)
        total_elements = app.num_chunks * channels * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        arr = np.frombuffer(raw, dtype=np.float32)
        assert np.all(arr < 0), "expected all-negative outputs (no ReLU applied)"
