"""Benchmark conv_first_layer: cycles + real MULT util.

Fixed shape (256x256x3 -> 128x128x16, stride 2), so there's only one config
-- unlike the other apps' benchmarks, this reports a single row. Compares
against a real PyTorch reference (tolerance-based, FP32 wide-vector mode).
Reports cycles and MULT-slot utilization read directly from ``state.stats``.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.conv.conv_first_layer.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.conv.conv_first_layer import (
    ConvFirstLayerApp,
    OUTPUT_BASE_ADDR,
    IN_ROWS,
    IN_CHANNELS,
    IN_COLS,
    OUT_ROWS,
    OUT_COLS,
    OUT_CHANNELS,
)
from ipu_apps.convolutions_universal.conv.conv_first_layer.test_conv_first_layer import (
    reference,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_first_layer.asm"

_TOL = 1e-2


def gen(seed: int = 42):
    rng = np.random.RandomState(seed)
    x = (rng.randn(IN_CHANNELS, IN_ROWS, IN_COLS) * 0.5).astype(np.float32)
    k = (rng.randn(OUT_CHANNELS, IN_CHANNELS, 3, 3) * 0.2).astype(np.float32)
    b = (rng.randn(OUT_CHANNELS) * 0.3).astype(np.float32)
    return x, k, b


def run_config(inst_file: Path):
    input_chw, kernel, bias = gen()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(input_chw.tobytes())

        app = ConvFirstLayerApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=kernel,
            bias=bias,
            output_path=None,
        )
        state, cycles = app.run(max_cycles=500_000_000)

        total_elements = OUT_ROWS * OUT_CHANNELS * OUT_COLS
        raw = state.xmem.read_address(OUTPUT_BASE_ADDR, total_elements * 4)
        out = np.frombuffer(raw, dtype=np.float32).reshape(OUT_ROWS, OUT_CHANNELS, OUT_COLS)
        actual = np.ascontiguousarray(out.transpose(1, 0, 2))

    expected = reference(kernel, input_chw, bias)
    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_first_layer.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        t0 = time.time()
        cycles, max_diff, mult_util = run_config(inst_file)
        elapsed = time.time() - t0

        rows_out = [BenchRow(
            label="256x256x3->128x128x16 (stride2)",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(max_diff < _TOL),
            elapsed_s=elapsed,
        )]

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("conv_first_layer", rows_out, out_path)


if __name__ == "__main__":
    main()
