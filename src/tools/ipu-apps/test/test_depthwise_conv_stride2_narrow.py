"""Self-contained tests for depthwise 3x3 stride-2 conv, cols in {16,32,64}
(packed-row two-stage app, FP32).

Runtime-generates random FP32 weights and inputs, runs the two-stage
emulator pipeline, and compares against a real
``torch.nn.functional.conv2d`` (groups=channels, stride=2) reference
(tolerance-based).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow import (
    DepthwiseConvStride2NarrowApp,
)

_TOL = 1e-2


def reference_stride2_narrow(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: depthwise 3x3 conv, stride 2, groups=channels."""
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


class TestDepthwiseConvStride2Narrow:

    @pytest.mark.parametrize(
        "channels,rows,cols,seed",
        [
            (2, 8, 64, 3),     # rows_per_chunk=2, minimal
            (4, 16, 32, 7),    # rows_per_chunk=4
            (3, 32, 16, 11),   # rows_per_chunk=8, odd channel count
        ],
    )
    def test_stride2_narrow(
        self, tmp_path: Path, channels: int, rows: int, cols: int, seed: int,
    ) -> None:
        rng = np.random.RandomState(seed)
        weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(channels, rows, cols) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvStride2NarrowApp(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            rows=rows,
            cols=cols,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)
        assert cycles > 0

        out_cols = cols // 2
        out_rows_per_chunk = 128 // out_cols
        num_out_groups = (rows // 2) // out_rows_per_chunk
        total_outputs = num_out_groups * channels
        raw = state.xmem.read_address(app.output_base_addr, total_outputs * 128 * 4)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(
            num_out_groups, channels, out_rows_per_chunk, out_cols,
        )
        out_rows = rows // 2
        actual = np.zeros((channels, out_rows, out_cols), dtype=np.float32)
        for og in range(num_out_groups):
            for ch in range(channels):
                for local_row in range(out_rows_per_chunk):
                    orow = og * out_rows_per_chunk + local_row
                    actual[ch, orow] = arr[og, ch, local_row]

        expected = reference_stride2_narrow(weights, input_chw)
        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for channels={channels} rows={rows} cols={cols}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_rejects_odd_rows(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((1, 5, 64), dtype=np.float32).tobytes())
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel=np.zeros((1, 3, 3), dtype=np.float32),
                output_path=None, rows=5, cols=64, channels=1,
            )

    def test_rejects_non_multiple_of_4_row_groups(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((1, 16, 16), dtype=np.float32).tobytes())
        # cols=16 -> rows_per_chunk=8 -> needs rows multiple of 32; 16 fails.
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel=np.zeros((1, 3, 3), dtype=np.float32),
                output_path=None, rows=16, cols=16, channels=1,
            )

    def test_rejects_invalid_cols(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((1, 8, 128), dtype=np.float32).tobytes())
        with pytest.raises(ValueError):
            DepthwiseConvStride2NarrowApp(
                input_path=input_file, kernel=np.zeros((1, 3, 3), dtype=np.float32),
                output_path=None, rows=8, cols=128, channels=1,
            )
