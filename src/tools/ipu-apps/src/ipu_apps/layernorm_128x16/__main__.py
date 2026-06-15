"""Debug runner for layernorm_128x16.

Usage::

    LAYERNORM_128X16_INST_BIN=/tmp/layernorm_128x16.bin \
    LAYERNORM_128X16_DATA_DIR=src/ipu_apps/layernorm_128x16/test_data_format \
    uv run python -m ipu_apps.layernorm_128x16
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm_128x16 import LayerNorm128x16App

_INST_BIN = Path(os.environ["LAYERNORM_128X16_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYERNORM_128X16_DATA_DIR"])

data_dir = _DATA_DIR / "wide_fp32"
app = LayerNorm128x16App(
    inst_path=_INST_BIN,
    input_path=data_dir / "input_x_fp32.bin",
    gamma_path=data_dir / "gamma_fp32.bin",
    beta_path=data_dir / "beta_fp32.bin",
    output_path="/tmp/layernorm_128x16_out.bin",
)
state = IpuState(
    wide_vector_debug=True,
    wide_vector_arithmetic=WideVectorArithmetic.FP32,
)
state, cycles = app.run(max_cycles=100_000, state=state)
print(f"Done in {cycles} cycles.")
print(state.stats.format_summary())
