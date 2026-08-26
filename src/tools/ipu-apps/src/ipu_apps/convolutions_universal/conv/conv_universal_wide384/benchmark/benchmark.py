"""Benchmark conv_universal_wide384: cycles + real MULT util.

Single-stage app (unlike the stride2 apps, no two-stage stats-composition
workaround needed) -- ``state.stats.mult_utilization`` is read directly.
Compares against a real PyTorch reference (tolerance-based, FP32
wide-vector mode).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_universal_wide384.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.conv.conv_universal_wide384 import (
    ConvUniversalWide384App,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


_TOL = 1e-2

# (width, rows, in_channels, out_channels) -- width multiple of 128, >=384;
# out_channels even (app requirement).
CONFIGS = [
    (384, 8, 1, 2),     # minimal cpr=3 case
    (384, 16, 3, 4),    # moderate spatial + channels
    (512, 8, 2, 4),     # cpr=4
    (384, 8, 16, 4),    # in_channels > FPB=14: exercises kernel-reload path
    (640, 4, 1, 2),     # cpr=5, wider than the primary target
]


def reference_conv_wide384(weights: np.ndarray, input_chw: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(input_chw).unsqueeze(0)
    w = torch.from_numpy(weights)
    return F.conv2d(x, w, padding=1).squeeze(0).numpy()


def run_config(width: int, rows: int, in_ch: int, out_ch: int, seed: int):
    rng = np.random.RandomState(seed)
    weights = (rng.randn(out_ch, in_ch, 3, 3) * 0.2).astype(np.float32)
    input_chw = (rng.randn(in_ch, rows, width) * 0.5).astype(np.float32)

    tmp = Path(tempfile.mkdtemp(prefix="conv_wide384_bench_"))
    input_file = tmp / "input.bin"
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

    total_elements = rows * out_ch * cpr * 128
    raw = state.xmem.read_address(app.output_base_addr, total_elements * 4)
    out = np.frombuffer(raw, dtype=np.float32).reshape(rows, out_ch, width)
    actual = np.ascontiguousarray(out.transpose(1, 0, 2))
    expected = reference_conv_wide384(weights, input_chw)

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    rows_out = []
    for width, rows, in_ch, out_ch in CONFIGS:
        seed = 100 + width + rows * 7 + in_ch * 13 + out_ch
        t0 = time.time()
        cycles, max_diff, mult_util = run_config(width, rows, in_ch, out_ch, seed)
        elapsed = time.time() - t0
        rows_out.append(BenchRow(
            label=f"{width}x{rows}x{in_ch}x{out_ch}",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(max_diff < _TOL),
            elapsed_s=elapsed,
        ))

    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("conv_universal_wide384", rows_out, out_path)


if __name__ == "__main__":
    main()
