"""Debug runner for qk_scores_256x36 (wide-vector FP32 mode).

Usage::

    QK_SCORES_256X36_INST_BIN=/tmp/qk_scores_256x36.bin \
    QK_SCORES_256X36_DATA_DIR=src/ipu_apps/qk_scores_256x36/test_data_format \
    uv run python -m ipu_apps.qk_scores_256x36
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_256x36 import QkScores256x36App

_INST_BIN = Path(os.environ["QK_SCORES_256X36_INST_BIN"])
_DATA_DIR = Path(os.environ["QK_SCORES_256X36_DATA_DIR"])

data_dir = _DATA_DIR / "wide_fp32"
app = QkScores256x36App(
    inst_path=_INST_BIN,
    query_path=data_dir / "query_fp32.bin",
    key_path=data_dir / "key_fp32.bin",
    output_path="/tmp/qk_scores_256x36_out.bin",
)
state = IpuState(
    wide_vector_debug=True,
    wide_vector_arithmetic=WideVectorArithmetic.FP32,
)
state, cycles = app.run(max_cycles=5_000_000, state=state)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
