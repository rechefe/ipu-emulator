"""Tests for the wide (W>=384) standard 3x3 convolution app, stride 1 (FP32).

Runtime-generates random FP32 weights and inputs, runs the emulator, and
compares against a real ``torch.nn.functional.conv2d`` reference
(tolerance-based, since IPU FP32 accumulation order differs from PyTorch's).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
    ConvUniversalWide384App,
)

_TOL = 1e-2


def reference_conv_wide384(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights)
    return F.conv2d(x, w, padding=1).squeeze(0).numpy()


class TestConvUniversalWide384:

    @pytest.mark.parametrize(
        "width,rows,in_ch,out_ch",
        [
            (384, 8, 1, 2),
            (384, 16, 2, 2),
            (384, 8, 3, 4),
        ],
    )
    def test_conv(
        self,
        tmp_path: Path,
        width: int,
        rows: int,
        in_ch: int,
        out_ch: int,
    ) -> None:
        rng = np.random.RandomState(42 + in_ch * 7 + out_ch + width)
        weights = (rng.randn(out_ch, in_ch, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(in_ch, rows, width) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = ConvUniversalWide384App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            width=width,
            rows=rows,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        cpr = width // 128
        max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_elements = rows * out_ch * cpr * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(rows, out_ch, width)
        actual = np.ascontiguousarray(out.transpose(1, 0, 2))
        expected = reference_conv_wide384(weights, input_chw)

        diff = np.abs(actual - expected).max()
        assert diff < _TOL, (
            f"max diff {diff:.3e} for width={width} rows={rows} in_ch={in_ch} out_ch={out_ch}\n"
            f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
            f"  expected[0,0,:8]: {expected[0, 0, :8]}"
        )

    def test_odd_out_channels_rejected(self, tmp_path: Path) -> None:
        input_file = tmp_path / "input.bin"
        input_file.write_bytes(np.zeros((1, 8, 384), dtype=np.float32).tobytes())
        weights = np.zeros((3, 1, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            ConvUniversalWide384App(
                input_path=input_file,
                kernel=weights,
                output_path=None,
                width=384,
                rows=8,
                in_channels=1,
                out_channels=3,
            )

    def test_width_512_bonus(self, tmp_path: Path) -> None:
        width, rows, in_ch, out_ch = 512, 4, 1, 2
        rng = np.random.RandomState(999)
        weights = (rng.randn(out_ch, in_ch, 3, 3) * 0.2).astype(np.float32)
        input_chw = (rng.randn(in_ch, rows, width) * 0.5).astype(np.float32)

        input_file = tmp_path / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = ConvUniversalWide384App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            width=width,
            rows=rows,
            in_channels=in_ch,
            out_channels=out_ch,
        )

        cpr = width // 128
        max_cyc = 200 * rows * out_ch * cpr * in_ch * 9 + 50_000
        state, cycles = app.run(max_cycles=max_cyc)
        assert cycles > 0

        total_elements = rows * out_ch * cpr * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(rows, out_ch, width)
        actual = np.ascontiguousarray(out.transpose(1, 0, 2))
        expected = reference_conv_wide384(weights, input_chw)

        diff = np.abs(actual - expected).max()
        assert diff < _TOL, f"max diff {diff:.3e}"
