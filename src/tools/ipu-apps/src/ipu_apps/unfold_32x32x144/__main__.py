"""Debug runner for unfold_32x32x144.

Usage::

    UNFOLD_32X32X144_INST_BIN=/tmp/unfold_32x32x144.bin \
    UNFOLD_32X32X144_DATA_DIR=src/ipu_apps/unfold_32x32x144/test_data_format \
    uv run python -m ipu_apps.unfold_32x32x144
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.unfold_32x32x144 import Unfold32x32x144App

_INST_BIN = Path(os.environ["UNFOLD_32X32X144_INST_BIN"])
_DATA_DIR = Path(os.environ["UNFOLD_32X32X144_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = Unfold32x32x144App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_int8.bin",
    output_path="/tmp/unfold_32x32x144_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=5_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
