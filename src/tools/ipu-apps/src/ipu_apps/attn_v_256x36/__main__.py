"""Debug runner for attn_v_256x36 (attn@V AGG kernel).

Usage::

    ATTN_V_256X36_INST_BIN=/tmp/attn_v_256x36.bin \
    ATTN_V_256X36_DATA_DIR=src/ipu_apps/attn_v_256x36/test_data_format \
    uv run python -m ipu_apps.attn_v_256x36
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.attn_v_256x36 import AttnV256x36App

_INST_BIN = Path(os.environ["ATTN_V_256X36_INST_BIN"])
_DATA_DIR = Path(os.environ["ATTN_V_256X36_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = AttnV256x36App(
    inst_path=_INST_BIN,
    p_path=dtype_dir / "p_int8.bin",
    v_path=dtype_dir / "v_int8.bin",
    output_path="/tmp/attn_v_256x36_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=20_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
