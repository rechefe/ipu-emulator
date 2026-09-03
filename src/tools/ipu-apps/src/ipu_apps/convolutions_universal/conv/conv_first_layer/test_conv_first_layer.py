"""Self-contained test for first-layer conv: 256x256x3 -> 128x128x16, stride 2 (FP32).

Generates random FP32 input/kernel/bias, assembles, runs, and compares
against a real ``torch.nn.functional.conv2d`` + bias -> ReLU reference
(tolerance-based, since IPU FP32 accumulation order differs from PyTorch's).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    OUTPUT_BASE_ADDR,
    IN_ROWS,
    IN_COLS,
    IN_CHANNELS,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
)

ASM_PATH = Path(__file__).resolve().parent / "conv_first_layer.asm"

_TOL = 1e-2


def reference(weights: np.ndarray, input_chw: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Real PyTorch reference: 3x3 stride-2 conv + bias -> ReLU."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights)
    b = torch.from_numpy(bias)
    return F.relu(F.conv2d(x, w, b, stride=2, padding=1)).squeeze(0).numpy()


def _gen(seed: int = 42):
    rng = np.random.RandomState(seed)
    x = (rng.randn(IN_CHANNELS, IN_ROWS, IN_COLS) * 0.5).astype(np.float32)
    k = (rng.randn(OUT_CHANNELS, IN_CHANNELS, 3, 3) * 0.2).astype(np.float32)
    b = (rng.randn(OUT_CHANNELS) * 0.3).astype(np.float32)
    return x, k, b


def test_256x256x3_to_128x128x16(tmp_path: Path) -> None:
    input_chw, kernel, bias = _gen()
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(input_chw.tobytes())

    inst_file = tmp_path / "prog.bin"
    assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

    app = ConvFirstLayerApp(
        inst_path=inst_file,
        input_path=input_file,
        kernel=kernel,
        bias=bias,
        output_path=None,
    )
    state, cycles = app.run(max_cycles=500_000_000)
    assert cycles > 0

    total_elements = OUT_ROWS * OUT_CHANNELS * OUT_COLS
    raw = state.xmem.read_address(OUTPUT_BASE_ADDR, total_elements * 4)
    out = np.frombuffer(raw, dtype=np.float32).reshape(OUT_ROWS, OUT_CHANNELS, OUT_COLS)
    actual = np.ascontiguousarray(out.transpose(1, 0, 2))  # -> [filter, row, col]
    expected = reference(kernel, input_chw, bias)

    diff = np.abs(actual - expected).max()
    assert diff < _TOL, (
        f"max diff {diff:.3e}\n"
        f"  actual[0,0,:8]:   {actual[0, 0, :8]}\n"
        f"  expected[0,0,:8]: {expected[0, 0, :8]}"
    )
