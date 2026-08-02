"""Benchmark conv_first_layer: cycles + real MULT util.

Fixed shape (256x256x3 -> 128x128x16, stride 2), so there's only one config
-- unlike the other apps' benchmarks, this reports a single row. Reports
cycles and MULT-slot utilization read directly from ``state.stats``.

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
    OUT_CHANNELS,
    OUT_ROW_GROUP,
)
from ipu_apps.convolutions_universal.conv.conv_first_layer.test_conv_first_layer import (
    reference,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = Path(__file__).resolve().parents[1] / "conv_first_layer.asm"


def gen(seed: int = 42):
    rng = np.random.RandomState(seed)
    x = rng.randint(-3, 4, size=IN_ROWS * IN_CHANNELS * IN_COLS, dtype=np.int8)
    k = rng.randint(-3, 4, size=(OUT_CHANNELS, IN_CHANNELS, 3, 3)).astype(np.int8)
    b = rng.randint(-5, 6, size=OUT_CHANNELS).astype(np.int8)
    return x.view(np.uint8).tobytes(), k, b


def run_config(inst_file: Path):
    input_bytes, kernel, bias = gen()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        input_file.write_bytes(input_bytes)

        app = ConvFirstLayerApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel=kernel,
            bias=bias,
            output_path=None,
        )
        state, cycles = app.run(max_cycles=500_000_000)

        total = OUT_ROWS * OUT_ROW_GROUP
        actual = bytes(state.xmem.read_address(OUTPUT_BASE_ADDR, total))

    expected = reference(input_bytes, kernel, bias)
    mismatches = sum(1 for i in range(len(expected)) if actual[i] != expected[i])
    return cycles, mismatches, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "conv_first_layer.bin"
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        t0 = time.time()
        cycles, mismatches, mult_util = run_config(inst_file)
        elapsed = time.time() - t0

        rows_out = [BenchRow(
            label="256x256x3->128x128x16 (stride2)",
            cycles=cycles,
            mult_utilization=mult_util,
            correct=(mismatches == 0),
            elapsed_s=elapsed,
        )]

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("conv_first_layer", rows_out, out_path)


if __name__ == "__main__":
    main()
