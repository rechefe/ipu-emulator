"""Debug runner for residual_add_256x144.

Usage::

    RESIDUAL_ADD_256X144_INST_BIN=/tmp/residual_add_256x144.bin \
    RESIDUAL_ADD_256X144_DATA_DIR=src/ipu_apps/residual_add_256x144/test_data_format \
    uv run python -m ipu_apps.residual_add_256x144
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.residual_add_256x144 import ResidualAdd256x144App

_INST_BIN = Path(os.environ["RESIDUAL_ADD_256X144_INST_BIN"])
_DATA_DIR = Path(os.environ["RESIDUAL_ADD_256X144_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = ResidualAdd256x144App(
    inst_path=_INST_BIN,
    input_a_path=dtype_dir / "input_a_int8.bin",
    input_b_path=dtype_dir / "input_b_int8.bin",
    output_path="/tmp/residual_add_256x144_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=100_000)
print(f"Done in {cycles} cycles.")
