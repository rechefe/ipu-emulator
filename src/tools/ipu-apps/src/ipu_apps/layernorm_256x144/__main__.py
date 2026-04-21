"""Debug runner for layernorm_256x144.

Usage::

    LAYERNORM_256X144_INST_BIN=/tmp/layernorm_256x144.bin \\
    LAYERNORM_256X144_DATA_DIR=src/ipu_apps/layernorm_256x144/test_data_format \\
    uv run python -m ipu_apps.layernorm_256x144
"""

from __future__ import annotations

import os
from pathlib import Path

from ipu_apps.layernorm_256x144 import LayerNorm256x144App

_INST_BIN = Path(os.environ["LAYERNORM_256X144_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYERNORM_256X144_DATA_DIR"])

dtype_dir = _DATA_DIR / "fp8_e4m3"
app = LayerNorm256x144App(
    inst_path=_INST_BIN,
    input_path=dtype_dir / "input_fp8_e4m3.bin",
    gamma_path=dtype_dir / "gamma_fp8_e4m3.bin",
    beta_path=dtype_dir  / "beta_fp8_e4m3.bin",
    output_path="/tmp/layernorm_256x144_out.bin",
)
state, cycles = app.run(max_cycles=5_000_000)
print(f"Done in {cycles} cycles.")
