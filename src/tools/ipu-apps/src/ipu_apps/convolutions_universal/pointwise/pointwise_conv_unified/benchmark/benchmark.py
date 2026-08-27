"""Benchmark pointwise_conv_unified across configs: cycles + correctness + efficiency.

Theoretical floor per output position: in_channels mults at 1 MAC/cyc.
Per row-group of 128 spatial positions: in_ch * out_ch mults across the
whole conv. Total mults = rows * cols * in_ch * out_ch / 128.

FP32 wide-vector mode: correctness compared against a real PyTorch reference
with a tolerance (IPU FP32 accumulation order differs from PyTorch's).

Usage::

    PYTHONPATH=... python -m ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified import (
    PointwiseConvUnifiedApp,
)
from ipu_apps.convolutions_universal.pointwise.pointwise_conv_unified.test_unified import (
    reference_pointwise,
)
from ipu_apps.convolutions_universal.benchmarking import BenchRow, print_and_write_table


ASM_PATH = (
    Path(__file__).resolve().parents[1] / "pointwise_conv_unified.asm"
)

_TOL = 1e-3

# (height, width, in_ch, out_ch). Sized to fit XMEM (2 MB).
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


def run_config(inst_file, height, width, in_ch, out_ch):
    rng = np.random.RandomState(42 + in_ch * 7 + out_ch + height + width)
    weights = (rng.randn(out_ch, in_ch) * 0.3).astype(np.float32)
    input_chw = (rng.randn(in_ch, height, width) * 0.5).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        input_file = tmp / "input.bin"
        kernel_file = tmp / "kernel.bin"
        output_file = tmp / "output.bin"
        input_file.write_bytes(input_chw.tobytes())
        kernel_file.write_bytes(weights.tobytes())

        app = PointwiseConvUnifiedApp(
            inst_path=inst_file,
            input_path=input_file,
            kernel_path=kernel_file,
            output_path=output_file,
            height=height, width=width, in_channels=in_ch, out_channels=out_ch,
        )
        max_cyc = 200 * in_ch * out_ch * app.row_groups + 100_000
        state, cycles = app.run(max_cycles=max_cyc)

        actual = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(
            out_ch, height, width
        )
        expected = reference_pointwise(weights, input_chw)

    max_diff = float(np.abs(actual - expected).max())
    return cycles, max_diff, state.stats.mult_utilization


def main() -> None:
    src = ASM_PATH.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        inst_file = Path(tmp) / "unified.bin"
        print("Assembling pointwise_conv_unified.asm ...", flush=True)
        assemble_to_bin_file(src, str(inst_file))

        rows_out = []
        for height, width, in_ch, out_ch in CONFIGS:
            t0 = time.time()
            cycles, max_diff, mult_util = run_config(inst_file, height, width, in_ch, out_ch)
            elapsed = time.time() - t0
            rows_out.append(BenchRow(
                label=f"{height}x{width} ic={in_ch} oc={out_ch}",
                cycles=cycles,
                mult_utilization=mult_util,
                correct=(max_diff < _TOL),
                elapsed_s=elapsed,
            ))

        out_path = Path(__file__).resolve().parent / "results.md"
        print_and_write_table("pointwise_conv_unified", rows_out, out_path)


if __name__ == "__main__":
    main()
