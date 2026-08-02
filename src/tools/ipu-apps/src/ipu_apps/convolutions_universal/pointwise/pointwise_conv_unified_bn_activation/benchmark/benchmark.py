"""Benchmark pointwise_conv_unified_bn_activation: cycles + real MULT util.

Reuses the app's own reference/packing helpers (test_bn_activation.py, next
to this app) so the reference math can't drift from the test suite's.
Reports cycles and MULT-slot utilization read directly from ``state.stats``.

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified_bn_activation.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified_bn_activation import (
    PointwiseConvUnifiedBnActivationApp,
)
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified_bn_activation.test_bn_activation import (
    ASM_PATH,
    pack_input,
    read_output,
    reference_pointwise_bn,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


# Mirrors pointwise_conv_unified's benchmark CONFIGS for comparison.
CONFIGS = [
    (128, 128,  64,  32),
    (128, 128,  32,  48),
    ( 64,  64, 128,  64),
    ( 64,  64,  64, 128),
    ( 32,  32, 256,  96),
    ( 32,  32,  96, 256),
    ( 16,  16, 384, 128),
    ( 16,  16, 128, 256),
]


def run_config(inst_file: Path, rows: int, cols: int, in_ch: int, out_ch: int):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + rows + cols)
    weights = rng.randint(-3, 4, size=(out_ch, in_ch), dtype=np.int8)
    input_chw = rng.randint(-3, 4, size=(in_ch, rows, cols), dtype=np.int8)
    bias = rng.randint(-40, 80, size=(out_ch,)).astype(np.int8)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        input_file.write_bytes(pack_input(input_chw, rows, cols))
        kernel_file.write_bytes(
            weights.reshape(out_ch * in_ch).view(np.uint8).tobytes()
        )

        app = PointwiseConvUnifiedBnActivationApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            bias=bias,
            output_path=None,
            dtype="INT8",
            rows=rows, cols=cols, in_channels=in_ch, out_channels=out_ch,
        )
        max_cyc = 50 * in_ch * out_ch * (rows * cols // 128) + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = read_output(state, rows, cols, out_ch)
        expected = reference_pointwise_bn(weights, input_chw, bias)

    mismatches = int(np.sum(actual != expected))
    return cycles, mismatches, state.stats.mult_utilization


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "unified_bn.bin"
        print("Assembling pointwise_conv_unified_bn_activation.asm ...", flush=True)
        assemble_to_bin_file(ASM_PATH.read_text(), str(inst_file))

        rows_out = []
        for rows, cols, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, mm, mult_util = run_config(inst_file, rows, cols, in_ch, out_ch)
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{rows}x{cols} ic={in_ch} oc={out_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(mm == 0),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("pointwise_conv_unified_bn_activation", rows_out, out_path)


if __name__ == "__main__":
    main()
