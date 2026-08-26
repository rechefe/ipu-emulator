"""Benchmark depthwise_conv_stride2_16: cycles + real MULT util.

Two-stage app (unmodified depthwise_conv_universal at the fixed 16x16
resolution, followed by an ACC.STRIDE joint row+col decimation pass that
packs 2 channels per output chunk), FP32 wide-vector mode. Compares against
a real PyTorch depthwise stride-2 conv2d reference (tolerance-based).
Reports cycles and MULT-slot utilization read directly from
``state.stats`` -- the reported utilization is dominated by stage 1 (the
conv), since stage 2 is a short, taps-free decimation pass.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_16.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_16 import (
    DepthwiseConvStride2_16App,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


_TOL = 1e-2

# channels must be even (two channels pack into one output chunk).
CONFIGS = [
    2,     # minimal case
    16,    # small channel count
    320,   # MobileViT-S's actual stage-5 shape
]


def reference_stride2(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


def run_config(channels: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(channels, 16, 16) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvStride2_16App(
            input_path=input_file,
            kernel=weights,
            output_path=output_file,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)

        actual = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(
            channels, 8, 8,
        )
        expected = reference_stride2(weights, input_chw)

        # Two-stage: stage 1 (conv) and stage 2 (decimate) run as separate
        # emulator invocations sharing one IpuState, so
        # state.stats.total_cycles gets overwritten with only stage 2's
        # count while mult_active_cycles accumulates across both. Recompute
        # utilization using the correct combined total (cycles == cycles1 +
        # cycles2, returned by DepthwiseConvStride2_16App.run()).
        mult_utilization = state.stats.mult_active_cycles / cycles if cycles else 0.0

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, mult_utilization


def main() -> None:
    rows_out = []
    for channels in CONFIGS:
        seed = 100 + channels
        t0 = time.time()
        cycles, max_diff, mult_util = run_config(channels, seed)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"16x16x{channels}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(max_diff < _TOL),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("depthwise_conv_stride2_16", rows_out, out_path)


if __name__ == "__main__":
    main()
