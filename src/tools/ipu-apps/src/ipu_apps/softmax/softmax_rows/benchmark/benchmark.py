"""Benchmark: softmax_rows across row-count configs.

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_rows.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_rows import SoftmaxRowsApp, LANES, ROW_BYTES
from ipu_apps.softmax.benchmarking import BenchRow, print_and_write_table

ASM_PATH = Path(__file__).resolve().parent.parent / "softmax_rows.asm"

CONFIGS = [8, 32, 128, 256, 500]


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def run_config(rows: int, asm_text: str) -> BenchRow:
    rng = np.random.RandomState(0)
    x = (rng.randn(rows, LANES) * 5.0).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_rows.bin"
        input_file = tmp / "input.bin"
        assemble_to_bin_file(asm_text, str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxRowsApp(
            inst_path=inst_file, input_path=input_file, output_path=None, rows=rows
        )
        state, cycles = app.run(max_cycles=4_000_000)
        raw = state.xmem.read_address(app.output_base, rows * ROW_BYTES)
        out = np.frombuffer(raw, dtype=np.float32).reshape(rows, LANES)

    max_err = float(np.abs(out - ref).max())
    return BenchRow(
        label=f"rows={rows}",
        cycles=cycles,
        cyc_per_row=cycles / rows,
        mult_utilization=state.stats.mult_utilization,
        acc_utilization=state.stats.acc_utilization,
        max_err=max_err,
        correct=max_err < 1e-4,
    )


def main() -> None:
    asm_text = ASM_PATH.read_text()
    rows = [run_config(rows, asm_text) for rows in CONFIGS]
    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("softmax_rows", rows, out_path)


if __name__ == "__main__":
    main()
