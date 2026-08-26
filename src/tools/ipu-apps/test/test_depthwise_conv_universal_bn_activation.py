"""Self-contained tests for depthwise 3x3 conv + folded bias + ReLU (FP32).

Runtime-generates random FP32 weights, inputs, and per-channel biases, runs
the emulator, and compares against a real ``torch.nn.functional.conv2d``
(groups=channels) + bias + ReLU reference (tolerance-based, since IPU FP32
accumulation order differs from PyTorch's).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation import (
    DepthwiseConvUniversalBnActivationApp,
    FPB,
)
from ipu_apps.convolutions_universal import unpack_output_chunked

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "ipu_apps" / "convolutions_universal"
    / "depthwise" / "depthwise_conv_universal_bn_activation"
    / "depthwise_conv_universal_bn_activation.asm"
)

_TOL = 1e-2


def reference_depthwise_bn_relu(
    weights: np.ndarray, input_chw: np.ndarray, bias: np.ndarray,
) -> np.ndarray:
    """Real PyTorch reference: depthwise 3x3 conv (groups=channels) + bias -> ReLU."""
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    b = torch.from_numpy(bias)
    return F.relu(F.conv2d(x, w, b, padding=1, groups=channels)).squeeze(0).numpy()


class TestDepthwiseConvUniversalBnActivation:

    @pytest.fixture(scope="class")
    def inst_file(self, tmp_path_factory) -> Path:
        tmp = tmp_path_factory.mktemp("dw_bn")
        inst_file = tmp / "depthwise_conv_universal_bn_activation.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))
        return inst_file

    @pytest.mark.parametrize(
        "channels,height,width",
        [
            (4, 16, 16),    # minimal, partial super-block
            (25, 16, 16),   # exactly one full FPB=25 super-block
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
    def test_depthwise_bn_relu(
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
        bias = (rng.randn(channels) * 0.3).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvUniversalBnActivationApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            bias=bias,
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
        expected = reference_depthwise_bn_relu(weights, input_chw, bias)

        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for channels={channels} {height}x{width}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_relu_zeros_negative_outputs(self, inst_file: Path, tmp_path: Path) -> None:
        """A strongly negative bias with zero weights must produce all-zero (ReLU) output."""
        height = width = 16
        channels = 4
        weights = np.zeros((channels, 3, 3), dtype=np.float32)  # conv contributes 0
        input_chw = np.ones((channels, height, width), dtype=np.float32)
        bias = np.array([-5.0, -0.1, -3.0, -1.0], dtype=np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvUniversalBnActivationApp(
            inst_path=inst_file, input_path=input_file, kernel=weights, bias=bias,
            output_path=None, height=height, width=width, channels=channels,
        )
        state, _ = app.run(max_cycles=2_000 * app.num_chunks * channels + 50_000)
        total_elements = app.num_chunks * channels * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        arr = np.frombuffer(raw, dtype=np.float32)
        assert np.all(arr == 0.0), "ReLU should zero all negative-bias outputs"
