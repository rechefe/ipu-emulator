"""Benchmark: softmax_columns_packed (down columns, width<=64, packed) across configs.

Usage::

    PYTHONPATH=... python -m ipu_apps.softmax.softmax_columns_packed.benchmark.benchmark
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp
from ipu_apps.softmax.benchmarking import BenchRow, print_and_write_table

ASM_PATH = Path(__file__).resolve().parent.parent / "softmax_columns_packed.asm"

# (rows, width) -- width in 1..64 (packed rows_per_vec = 128/width lanes).
CONFIGS = [(64, 8), (64, 16), (100, 32), (128, 64)]


def reference_softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=0, keepdims=True))
    return z / z.sum(axis=0, keepdims=True)


def run_config(rows: int, width: int, asm_text: str) -> BenchRow:
    rng = np.random.RandomState(0)
    x = (rng.randn(rows, width) * 3.0).astype(np.float32)
    ref = reference_softmax(x)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inst_file = tmp / "softmax_columns_packed.bin"
        input_file = tmp / "input.bin"
        output_file = tmp / "output.bin"
        assemble_to_bin_file(asm_text, str(inst_file))
        input_file.write_bytes(x.tobytes())

        app = SoftmaxColumnsPackedApp(
            inst_path=inst_file, input_path=input_file, output_path=output_file,
            rows=rows, width=width,
        )
        state, cycles = app.run(max_cycles=8_000_000)
        out = np.frombuffer(output_file.read_bytes(), dtype=np.float32).reshape(rows, width)

    max_err = float(np.abs(out - ref).max())
    return BenchRow(
        label=f"rows={rows},width={width}",
        cycles=cycles,
        cyc_per_row=cycles / rows,
        mult_utilization=state.stats.mult_utilization,
        acc_utilization=state.stats.acc_utilization,
        max_err=max_err,
        correct=max_err < 1e-4,
    )


def main() -> None:
    asm_text = ASM_PATH.read_text()
    rows = [run_config(rows, width, asm_text) for rows, width in CONFIGS]
    out_path = Path(__file__).resolve().parent / "results.md"
    print_and_write_table("softmax_columns_packed", rows, out_path)


if __name__ == "__main__":
    main()
