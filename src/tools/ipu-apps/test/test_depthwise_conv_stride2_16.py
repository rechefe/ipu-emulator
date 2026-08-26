"""Self-contained tests for depthwise 3x3 stride-2 conv, the fixed 16x16
shape (two-stage app, FP32).

Runtime-generates random FP32 weights and inputs, runs the two-stage
emulator pipeline, and compares against a real
``torch.nn.functional.conv2d`` (groups=channels, stride=2) reference
(tolerance-based).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_16 import (
    DepthwiseConvStride2_16App,
)

_TOL = 1e-2


def reference_stride2(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: depthwise 3x3 conv, stride 2, groups=channels."""
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


class TestDepthwiseConvStride2_16:

    @pytest.mark.parametrize(
        "channels,seed",
        [
            (2, 3),     # minimal case: one output chunk, one channel pair
            (4, 7),     # two channel pairs
            (320, 13),  # MobileViT-S's actual stage-5 shape
        ],
    )
    def test_stride2(self, tmp_path: Path, channels: int, seed: int) -> None:
        rng = np.random.RandomState(seed)
        weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(channels, 16, 16) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        output_file = tmp_path / "output.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvStride2_16App(
            input_path=input_file,
            kernel=weights,
            output_path=output_file,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)
        assert cycles > 0

        actual = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(
            channels, 8, 8,
        )
        expected = reference_stride2(weights, input_chw)
        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for channels={channels}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_rejects_odd_channels(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((3, 16, 16), dtype=np.float32).tobytes())
        with pytest.raises(ValueError):
            DepthwiseConvStride2_16App(
                input_path=input_file,
                kernel=np.zeros((3, 3, 3), dtype=np.float32),
                output_path=None, channels=3,
            )
