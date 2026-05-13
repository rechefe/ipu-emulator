"""Debug runner for rf_feedback.

Usage::

    RF_FEEDBACK_INST_BIN=/tmp/rf_feedback.bin \\
    RF_FEEDBACK_DATA_DIR=src/ipu_apps/rf_feedback/test_data_format \\
    uv run python -m ipu_apps.rf_feedback
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.rf_feedback import RfFeedbackApp

_INST_BIN = Path(os.environ["RF_FEEDBACK_INST_BIN"])
_DATA_DIR = Path(os.environ["RF_FEEDBACK_DATA_DIR"])

dtype_dir = _DATA_DIR / "int8"
app = RfFeedbackApp(
    inst_path=_INST_BIN,
    scalar_path=dtype_dir / "scalar_int8.bin",
    data_path=dtype_dir / "data_int8.bin",
    output_path="/tmp/rf_feedback_out.bin",
    dtype="INT8",
)
state, cycles = app.run(max_cycles=10_000)
print(f"Done in {cycles} cycles.")
