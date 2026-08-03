"""Debug runner for unfold_16x16x192.

Usage::

    UNFOLD_16X16X192_INST_BIN=/tmp/unfold_16x16x192.bin \
    UNFOLD_16X16X192_DATA_DIR=src/ipu_apps/unfold_16x16x192/test_data_format \
    uv run python -m ipu_apps.unfold_16x16x192
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.unfold_16x16x192 import Unfold16x16x192App

_INST_BIN = Path(os.environ["UNFOLD_16X16X192_INST_BIN"])
_DATA_DIR = Path(os.environ["UNFOLD_16X16X192_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = Unfold16x16x192App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_int8.bin",
    output_path="/tmp/unfold_16x16x192_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=5_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
