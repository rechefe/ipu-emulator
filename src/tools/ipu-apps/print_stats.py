"""Print RunStats for all apps on ZDlinear (INT8, one run each).

Usage from src/tools/ipu-apps/:
    uv run python print_stats.py /tmp
where /tmp contains the assembled .bin files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ipu_apps.matmul_128x128 import MatMul128x128App
from ipu_apps.matmul_128x64x128 import MatMul128x64x128App
from ipu_apps.matmul_128x64x64 import MatMul128x64x64App
from ipu_apps.matmul_64x64x64 import MatMul64x64x64App
from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
from ipu_apps.matmul_288x144_x128 import MatMul288x144x128App
from ipu_apps.matmul_432x144_x128 import MatMul432x144x128App
from ipu_apps.matmul_144x288_x128 import MatMul144x288x128App
from ipu_apps.unfold_32x32x144 import Unfold32x32x144App

BIN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
DATA_ROOT = Path("src/ipu_apps")

# (name, AppClass, has_weights, max_cycles)
APPS = [
    ("matmul_128x128",     MatMul128x128App,     True,  2_000_000),
    ("matmul_128x64x128",  MatMul128x64x128App,  True,  2_000_000),
    ("matmul_128x64x64",   MatMul128x64x64App,   True,  2_000_000),
    ("matmul_64x64x64",    MatMul64x64x64App,    True,  2_000_000),
    ("matmul_144x144_x128",MatMul144x144x128App, True,  5_000_000),
    ("matmul_288x144_x128",MatMul288x144x128App, True,  5_000_000),
    ("matmul_432x144_x128",MatMul432x144x128App, True, 10_000_000),
    ("matmul_144x288_x128",MatMul144x288x128App, True, 10_000_000),
    ("unfold_32x32x144",   Unfold32x32x144App,   False, 5_000_000),
]

for name, AppClass, has_weights, max_cycles in APPS:
    data_dir = DATA_ROOT / name / "test_data_format" / "int8"
    kwargs = dict(
        inst_path=BIN_DIR / f"{name}.bin",
        input_path=data_dir / "input_int8.bin",
        output_path=None,
        dtype="INT8",
    )
    if has_weights:
        kwargs["weights_path"] = data_dir / "weights_int8.bin"
    app = AppClass(**kwargs)
    state, cycles = app.run(max_cycles=max_cycles)
    print(f"\n=== {name} ===")
    print(state.stats.format_summary())
