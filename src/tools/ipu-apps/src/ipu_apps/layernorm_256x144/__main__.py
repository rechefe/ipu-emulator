"""Debug runner for layernorm_256x144.

Usage::

    LAYERNORM_256X144_INST_BIN=/tmp/layernorm_256x144.bin \\
    LAYERNORM_256X144_DATA_DIR=src/ipu_apps/layernorm_256x144/test_data_format \\
    uv run python -m ipu_apps.layernorm_256x144
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm_256x144 import LayerNorm256x144App

_INST_BIN = Path(os.environ["LAYERNORM_256X144_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYERNORM_256X144_DATA_DIR"])

data_dir = _DATA_DIR / "wide_fp32"
app = LayerNorm256x144App(
    inst_path=_INST_BIN,
    input_path=data_dir / "input_x_fp32.bin",
    gamma_path=data_dir / "gamma_fp32.bin",
    beta_path=data_dir  / "beta_fp32.bin",
    output_path="/tmp/layernorm_256x144_out.bin",
)
state = IpuState(
    wide_vector_debug=True,
    wide_vector_arithmetic=WideVectorArithmetic.FP32,
)
state, cycles = app.run(max_cycles=5_000_000, state=state)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
