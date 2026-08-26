"""Benchmark depthwise_conv_stride2_narrow: cycles + real MULT util.

Two-stage depthwise 3x3 stride-2 conv, cols in {16, 32, 64} (packed
multi-row-per-chunk layout), FP32 wide-vector mode. Compares against a real
PyTorch depthwise stride-2 conv2d reference (tolerance-based). Reports
cycles and MULT-slot utilization read directly from ``state.stats`` (the
emulator's live per-cycle occupancy counter) for stage 1 + stage 2 combined.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_narrow import (
    DepthwiseConvStride2NarrowApp,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


_TOL = 1e-2

# (rows, cols, channels) — rows must be a multiple of 4*rows_per_chunk,
# rows_per_chunk = 128/cols: cols=64 -> mult of 8, cols=32 -> mult of 16,
# cols=16 -> mult of 32.
CONFIGS = [
    (8, 64, 2),         # cols=64, minimal
    (8, 64, 16),        # cols=64, more channels
    (16, 32, 4),        # cols=32, small
    (16, 32, 24),       # cols=32, larger channel count
    (32, 16, 3),        # cols=16, odd/non-packing-multiple channel count
    (32, 16, 16),       # cols=16, larger channel count
]


def reference_stride2_narrow(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    channels = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights).unsqueeze(1)
    return F.conv2d(x, w, padding=1, stride=2, groups=channels).squeeze(0).numpy()


def run_config(rows: int, cols: int, channels: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = (rng.randn(channels, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(channels, rows, cols) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
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

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    rows_out = []
    for i, (rows, cols, channels) in enumerate(CONFIGS):
        t0 = time.time()
        cycles, max_diff, mult_util = run_config(rows, cols, channels, seed=100 + i)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{rows}x{cols}x{channels}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(max_diff < _TOL),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("depthwise_conv_stride2_narrow", rows_out, out_path)


if __name__ == "__main__":
    main()
