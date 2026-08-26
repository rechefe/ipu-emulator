"""Benchmark depthwise_conv_stride2_128: cycles + real MULT util.

Two-stage app (unmodified depthwise_conv_universal at full resolution,
cols=128, followed by an ACC.STRIDE column-decimation pass), FP32
wide-vector mode. Compares against a real PyTorch depthwise stride-2 conv2d
reference (tolerance-based). Reports cycles and MULT-slot utilization read
directly from ``state.stats`` (the emulator's live per-cycle occupancy
counter) -- the reported utilization is dominated by stage 1 (the conv),
since stage 2 is a short, taps-free decimation pass.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128.benchmark.benchmark
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


_TOL = 1e-2

# (rows, channels) -- rows must be even and >=4, rows/2 (out_rows) must also
# be even (two output rows pack into one chunk per stage 2 word).
CONFIGS = [
    (32, 8),     # small spatial, few channels
    (64, 16),    # multi-chunk, mid channels
    (128, 16),   # large spatial, primary benchmark
    (32, 64),    # many channels, small spatial
    (128, 32),   # exercises dynamic region sizing at larger rows*channels
]


def reference_stride2(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


def run_config(rows: int, channels: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(channels, rows, 128) * 0.5).astype(np.float32)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = DepthwiseConvStride2_128App(
            input_path=input_file,
            kernel=weights,
            output_path=None,
            rows=rows,
            channels=channels,
        )
        state, cycles = app.run(max_cycles=2_000_000)

        out_rows = rows // 2
        num_row_pairs = out_rows // 2
        total_outputs = num_row_pairs * channels
        raw = state.xmem.read_address(app.output_base_addr, total_outputs * 128 * 4)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(num_row_pairs, channels, 2, 64)
        actual = np.zeros((channels, out_rows, 64), dtype=np.float32)
        for rp in range(num_row_pairs):
            for ch in range(channels):
                actual[ch, 2 * rp] = arr[rp, ch, 0]
                actual[ch, 2 * rp + 1] = arr[rp, ch, 1]

        expected = reference_stride2(weights, input_chw)

        # Two-stage: stage 1 (conv) and stage 2 (decimate) run as separate
        # emulator invocations sharing one IpuState, so
        # state.stats.total_cycles gets overwritten with only stage 2's
        # count while mult_active_cycles accumulates across both. Recompute
        # utilization using the correct combined total (cycles == cycles1 +
        # cycles2, returned by DepthwiseConvStride2_128App.run()).
        mult_utilization = state.stats.mult_active_cycles / cycles if cycles else 0.0

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, mult_utilization


def main() -> None:
    rows_out = []
    for rows, channels in CONFIGS:
        seed = 100 + rows * 7 + channels
        t0 = time.time()
        cycles, max_diff, mult_util = run_config(rows, channels, seed)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{rows}x{channels}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(max_diff < _TOL),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("depthwise_conv_stride2_128", rows_out, out_path)


if __name__ == "__main__":
    main()
