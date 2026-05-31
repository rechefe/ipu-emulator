"""Print RunStats for all transformer matmul apps (INT8, one run each).

Usage from src/tools/ipu-apps/:
    uv run python print_stats.py /tmp
where /tmp contains the assembled .bin files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
from ipu_apps.matmul_288x144_x128 import MatMul288x144x128App
from ipu_apps.matmul_432x144_x128 import MatMul432x144x128App
from ipu_apps.matmul_144x288_x128 import MatMul144x288x128App

BIN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
DATA_ROOT = Path("src/ipu_apps")

APPS = [
    ("matmul_144x144_x128", MatMul144x144x128App),
    ("matmul_288x144_x128", MatMul288x144x128App),
    ("matmul_432x144_x128", MatMul432x144x128App),
    ("matmul_144x288_x128", MatMul144x288x128App),
]

for name, AppClass in APPS:
    data_dir = DATA_ROOT / name / "test_data_format" / "int8"
    app = AppClass(
        inst_path=BIN_DIR / f"{name}.bin",
        input_path=data_dir / "input_int8.bin",
        weights_path=data_dir / "weights_int8.bin",
        output_path=None,
        dtype="INT8",
    )
    state, cycles = app.run(max_cycles=10_000_000)
    print(f"\n=== {name} ===")
    print(state.stats.format_summary())
