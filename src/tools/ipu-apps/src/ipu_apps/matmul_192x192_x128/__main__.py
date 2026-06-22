"""Debug runner for matmul_192x192_x128."""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.matmul_192x192_x128 import MatMul192x192x128App

_INST_BIN = Path(os.environ["MATMUL_192X192_X128_INST_BIN"])
_DATA_DIR = Path(os.environ["MATMUL_192X192_X128_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"

app = MatMul192x192x128App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_int8.bin",
    weights_path=dtype_dir / "weights_int8.bin",
    output_path="/tmp/matmul_192x192_x128_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=10_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
