"""Benchmark: softmax_rows_long (N>128, N%128 != 0) across configs.

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_rows_long.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp
from ipu_apps.softmax.benchmarking import BenchRow, print_and_write_table

ASM_PATH = Path(__file__).resolve().parent.parent / "softmax_rows_long.asm"

# (rows, n) -- n > 128, n % 128 != 0.
# (rows, n). The last two cross a 128-row group boundary, where the kernel
# re-runs all four passes on the next group -- kept here so the per-group
# overhead stays visible in cyc/row.
CONFIGS = [(8, 200), (8, 300), (16, 500), (4, 1000), (200, 200), (300, 129)]


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def run_config(rows: int, n: int, asm_text: str) -> BenchRow:
    rng = np.random.RandomState(0)
    x = (rng.randn(rows, n) * 5.0).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_rows_long.bin"
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        assemble_to_bin_file(asm_text, str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxRowsLongApp(
            inst_path=inst_file, input_path=input_file, output_path=output_file, rows=rows, n=n
        )
        state, cycles = app.run(max_cycles=8_000_000)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(rows, n)

    max_err = float(np.abs(out - ref).max())
    return BenchRow(
        label=f"rows={rows},n={n}",
        cycles=cycles,
        cyc_per_row=cycles / rows,
        mult_utilization=state.stats.mult_utilization,
        acc_utilization=state.stats.acc_utilization,
        max_err=max_err,
        correct=max_err < 1e-4,
    )


def main() -> None:
    asm_text = ASM_PATH.read_text()
    rows = [run_config(rows, n, asm_text) for rows, n in CONFIGS]
    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("softmax_rows_long", rows, out_path)


if __name__ == "__main__":
    main()
