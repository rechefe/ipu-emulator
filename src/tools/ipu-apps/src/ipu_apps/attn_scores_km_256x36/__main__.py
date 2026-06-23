"""Debug runner for attn_scores_km_256x36.

Usage::

    ATTN_SCORES_KM_256X36_INST_BIN=/tmp/attn_scores_km_256x36.bin \
    ATTN_SCORES_KM_256X36_DATA_DIR=src/ipu_apps/attn_scores_km_256x36/test_data_format \
    uv run python -m ipu_apps.attn_scores_km_256x36
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.attn_scores_km_256x36 import AttnScoresKM256x36App

_INST_BIN = Path(os.environ["ATTN_SCORES_KM_256X36_INST_BIN"])
_DATA_DIR = Path(os.environ["ATTN_SCORES_KM_256X36_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = AttnScoresKM256x36App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_int8.bin",
    weights_path=dtype_dir / "weights_int8.bin",
    output_path="/tmp/attn_scores_km_256x36_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=5_000_000)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
