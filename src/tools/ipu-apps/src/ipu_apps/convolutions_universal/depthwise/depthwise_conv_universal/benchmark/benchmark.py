"""Benchmark depthwise_conv_universal: cycles + real MULT util.

No bias / no activation (the BN twin has the folded-bias version). Compares
output to a real PyTorch depthwise conv2d reference (tolerance-based, FP32
wide-vector mode). Reports cycles and MULT-slot utilization read directly
from ``state.stats`` (the emulator's live per-cycle occupancy counter).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal.benchmark.benchmark
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    FPB,
)
from ipu_apps.convolutions_universal import unpack_output_chunked
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = (
    Path(__file__).resolve().parents[1] / "depthwise_conv_universal.asm"
)

_TOL = 1e-2

# (height, width, channels) — spatial sizes + FPB=28 boundary cases.
CONFIGS = [
    (16, 16, 8),       # partial single block
    (16, 16, 28),      # exactly 1 full FPB=28 block
    (16, 16, 29),      # 1 full + 1-channel partial
    (32, 32, 16),      # multi-chunk, partial block
    (32, 32, 32),      # multi-chunk, 1 full + partial
    (32, 32, 56),      # exactly 2 full blocks
    (64, 64, 32),      # primary benchmark — large spatial
    (64, 64, 64),      # large spatial, multiple blocks
    (32, 32, 96),      # many channels
    (16, 16, 40),      # two blocks, small spatial
]


def reference_depthwise(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)  # [channels, 1, 3, 3]
    return F.conv2d(x, w, padding=1, groups=channels).squeeze(0).numpy()


def run_config(inst_file: Path, height: int, width: int, channels: int):
    rng = np.random.RandomState(42 + channels * 7 + height + width)
    weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(channels, height, width) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvUniversalApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=weights,
            output_path=None,
            height=height, width=width, channels=channels,
        )

        max_cyc = 2000 * app.num_chunks * channels * math.ceil(channels / FPB) + 50_000
        state, cycles = app.run(max_cycles=max_cyc)

        total_elements = app.num_chunks * channels * 128
        raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
        padded_out = unpack_output_chunked(raw, channels, app.rows, app.cols)
        actual = padded_out[:, :height, :width]
        expected = reference_depthwise(weights, input_chw)

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "depthwise_conv_universal.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for height, width, channels in CONFIGS:
            t0 = time.time()
            cycles, max_diff, mult_util = run_config(
                inst_file, height, width, channels
            )
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{height}x{width}x{channels}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(max_diff < _TOL),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("depthwise_conv_universal", rows_out, out_path)


if __name__ == "__main__":
    main()
