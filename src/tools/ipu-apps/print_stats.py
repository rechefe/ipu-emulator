"""Print RunStats for all apps on ZDlinear (INT8 for matmuls, FP32 for layernorms).

Usage from src/tools/ipu-apps/:
    uv run python print_stats.py /tmp
where /tmp contains the assembled .bin files.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_128x128 import MatMul128x128App
from ipu_apps.matmul_128x64x128 import MatMul128x64x128App
from ipu_apps.matmul_128x64x64 import MatMul128x64x64App
from ipu_apps.matmul_64x64x64 import MatMul64x64x64App
from ipu_apps.matmul_144x144_x128 import MatMul144x144x128App
from ipu_apps.matmul_288x144_x128 import MatMul288x144x128App
from ipu_apps.matmul_432x144_x128 import MatMul432x144x128App
from ipu_apps.matmul_144x288_x128 import MatMul144x288x128App
from ipu_apps.unfold_32x32x144 import Unfold32x32x144App
from ipu_apps.layernorm_128x16 import LayerNorm128x16App
from ipu_apps.layernorm_256x144 import LayerNorm256x144App

BIN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
DATA_ROOT = Path("src/ipu_apps")

# (name, AppClass, kwargs_fn, max_cycles)
def _matmul_kwargs(name, has_weights, bin_dir, data_root):
    data_dir = data_root / name / "test_data_format" / "int8"
    kw = dict(inst_path=bin_dir / f"{name}.bin", input_path=data_dir / "input_int8.bin",
              output_path=None, dtype="INT8")
    if has_weights:
        kw["weights_path"] = data_dir / "weights_int8.bin"
    return kw, None  # (kwargs, custom_state)

def _layernorm_kwargs(name, bin_dir, data_root):
    data_dir = data_root / name / "test_data_format" / "wide_fp32"
    kw = dict(inst_path=bin_dir / f"{name}.bin", input_path=data_dir / "input_x_fp32.bin",
              gamma_path=data_dir / "gamma_fp32.bin", beta_path=data_dir / "beta_fp32.bin",
              output_path=None)
    state = IpuState(wide_vector_debug=True, wide_vector_arithmetic=WideVectorArithmetic.FP32)
    return kw, state

APPS = [
    ("matmul_128x128",      MatMul128x128App,      lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  2_000_000),
    ("matmul_128x64x128",   MatMul128x64x128App,   lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  2_000_000),
    ("matmul_128x64x64",    MatMul128x64x64App,    lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  2_000_000),
    ("matmul_64x64x64",     MatMul64x64x64App,     lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  2_000_000),
    ("matmul_144x144_x128", MatMul144x144x128App,  lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  5_000_000),
    ("matmul_288x144_x128", MatMul288x144x128App,  lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT),  5_000_000),
    ("matmul_432x144_x128", MatMul432x144x128App,  lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT), 10_000_000),
    ("matmul_144x288_x128", MatMul144x288x128App,  lambda n: _matmul_kwargs(n, True,  BIN_DIR, DATA_ROOT), 10_000_000),
    ("unfold_32x32x144",    Unfold32x32x144App,    lambda n: _matmul_kwargs(n, False, BIN_DIR, DATA_ROOT),  5_000_000),
    ("layernorm_128x16",    LayerNorm128x16App,    lambda n: _layernorm_kwargs(n, BIN_DIR, DATA_ROOT),      100_000),
    ("layernorm_256x144",   LayerNorm256x144App,   lambda n: _layernorm_kwargs(n, BIN_DIR, DATA_ROOT),    5_000_000),
]

for name, AppClass, kwargs_fn, max_cycles in APPS:
    kwargs, custom_state = kwargs_fn(name)
    app = AppClass(**kwargs)
    run_kw = dict(max_cycles=max_cycles)
    if custom_state is not None:
        run_kw["state"] = custom_state
    state, cycles = app.run(**run_kw)
    print(f"\n=== {name} ===")
    print(state.stats.format_summary())
