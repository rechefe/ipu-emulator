"""Self-contained tests for depthwise 3x3 stride-2 conv, cols=128 (two-stage app, FP32).

Runtime-generates random FP32 weights and inputs, runs the two-stage
emulator pipeline, and compares against a real
``torch.nn.functional.conv2d`` (groups=channels, stride=2) reference
(tolerance-based).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
)
from ipu_apps.convolutions_universal import unpack_output_chunked

_TOL = 1e-2


def reference_stride2(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: depthwise 3x3 conv, stride 2, groups=channels."""
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


class TestDepthwiseConvStride2_128:

    @pytest.mark.parametrize(
        "channels,rows,seed",
        [
            (2, 8, 3),      # minimal case
            (4, 16, 7),     # exercises the ch_loop cross-word row advance twice
            (3, 32, 11),    # odd channel count, larger spatial extent
            (32, 128, 13),  # rows*channels large: exercises dynamic region sizing
        ],
    )
    def test_stride2(
        self, tmp_path: Path, channels: int, rows: int, seed: int,
    ) -> None:
        rng = np.random.RandomState(seed)
        weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(channels, rows, 128) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvStride2_128App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            rows=rows,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)
        assert cycles > 0

        out_rows = rows // 2
        num_row_pairs = out_rows // 2
        total_outputs = num_row_pairs * channels
        raw = state.xmem.read_address(app.output_base_addr, total_outputs * 128 * 4)
        # Two output rows per chunk, packed as [row_pair, channel, 128 elements].
        arr = np.frombuffer(raw, dtype=np.float32).reshape(num_row_pairs, channels, 2, 64)
        actual = np.zeros((channels, out_rows, 64), dtype=np.float32)
        for rp in range(num_row_pairs):
            for ch in range(channels):
                actual[ch, 2 * rp] = arr[rp, ch, 0]
                actual[ch, 2 * rp + 1] = arr[rp, ch, 1]

        expected = reference_stride2(weights, input_chw)
        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for channels={channels} rows={rows}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_rejects_odd_rows(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((1, 5, 128), dtype=np.float32).tobytes())
        with pytest.raises(ValueError):
            DepthwiseConvStride2_128App(
                input_path=input_file,
                kernel=np.zeros((1, 3, 3), dtype=np.float32),
                output_path=None, rows=5, channels=1,
            )
