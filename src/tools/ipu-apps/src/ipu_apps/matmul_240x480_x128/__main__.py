"""Debug runner for matmul_240x480_x128."""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.matmul_240x480_x128 import MatMul240x480x128App

_INST_BIN = Path(os.environ["MATMUL_240X480_X128_INST_BIN"])
_DATA_DIR = Path(os.environ["MATMUL_240X480_X128_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"

app = MatMul240x480x128App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_int8.bin",
    weights_path=dtype_dir / "weights_int8.bin",
    output_path="/tmp/matmul_240x480_x128_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=10_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
