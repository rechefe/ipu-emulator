"""Print RunStats (incl. mult utilization) for ALL 25 kernels on ZDlinear.

Run from src/tools/ipu-apps/ after assembling every .bin into a bin dir:

    bash assemble_all.sh /tmp/bins
    uv run python print_stats_all.py /tmp/bins

Matmuls/attn use INT8; layernorms and qk_scores use wide-vector FP32 debug.
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
from ipu_apps.matmul_192x192_x128 import MatMul192x192x128App
from ipu_apps.matmul_192x384_x128 import MatMul192x384x128App
from ipu_apps.matmul_240x240_x128 import MatMul240x240x128App
from ipu_apps.matmul_240x480_x128 import MatMul240x480x128App
from ipu_apps.matmul_384x192_x128 import MatMul384x192x128App
from ipu_apps.matmul_480x240_x128 import MatMul480x240x128App
from ipu_apps.matmul_576x192_x128 import MatMul576x192x128App
from ipu_apps.matmul_720x240_x128 import MatMul720x240x128App
from ipu_apps.unfold_32x32x144 import Unfold32x32x144App
from ipu_apps.residual_add_256x144 import ResidualAdd256x144App
from ipu_apps.layernorm_128x16 import LayerNorm128x16App
from ipu_apps.layernorm_256x144 import LayerNorm256x144App
from ipu_apps.attn_scores_km_256x36 import AttnScoresKM256x36App
from ipu_apps.attn_v_256x36 import AttnV256x36App
from ipu_apps.attn_v_bcast_36 import AttnVBcast36App
from ipu_apps.qk_scores_256x36 import QkScores256x36App

BIN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/bins")
DATA_ROOT = Path("src/ipu_apps")


def _fmt(root, name):
    return root / name / "test_data_format"


def matmul(App, has_weights=True):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "int8"
        kw = dict(inst_path=BIN_DIR / f"{name}.bin",
                  input_path=d / "input_int8.bin",
                  output_path=None, dtype="INT8")
        if has_weights:
            kw["weights_path"] = d / "weights_int8.bin"
        return App(**kw), None
    return build


def attn_pv(App):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "int8"
        app = App(inst_path=BIN_DIR / f"{name}.bin",
                  p_path=d / "p_int8.bin", v_path=d / "v_int8.bin",
                  output_path=None, dtype="INT8")
        return app, None
    return build


def attn_scores(App):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "int8"
        app = App(inst_path=BIN_DIR / f"{name}.bin",
                  input_path=d / "input_int8.bin",
                  weights_path=d / "weights_int8.bin",
                  output_path=None, dtype="INT8")
        return app, None
    return build


def qk(App):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "wide_fp32"
        app = App(inst_path=BIN_DIR / f"{name}.bin",
                  query_path=d / "query_fp32.bin",
                  key_path=d / "key_fp32.bin", output_path=None)
        state = IpuState(wide_vector_debug=True,
                         wide_vector_arithmetic=WideVectorArithmetic.FP32)
        return app, state
    return build


def residual(App):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "int8"
        app = App(inst_path=BIN_DIR / f"{name}.bin",
                  input_a_path=d / "input_a_int8.bin",
                  input_b_path=d / "input_b_int8.bin",
                  output_path=None, dtype="INT8")
        return app, None
    return build


def layernorm(App):
    def build(name):
        d = _fmt(DATA_ROOT, name) / "wide_fp32"
        app = App(inst_path=BIN_DIR / f"{name}.bin",
                  input_path=d / "input_x_fp32.bin",
                  gamma_path=d / "gamma_fp32.bin",
                  beta_path=d / "beta_fp32.bin", output_path=None)
        state = IpuState(wide_vector_debug=True,
                         wide_vector_arithmetic=WideVectorArithmetic.FP32)
        return app, state
    return build


# (name, builder, max_cycles)
APPS = [
    ("matmul_128x128",      matmul(MatMul128x128App),       2_000_000),
    ("matmul_128x64x128",   matmul(MatMul128x64x128App),    2_000_000),
    ("matmul_128x64x64",    matmul(MatMul128x64x64App),     2_000_000),
    ("matmul_64x64x64",     matmul(MatMul64x64x64App),      2_000_000),
    ("matmul_144x144_x128", matmul(MatMul144x144x128App),   5_000_000),
    ("matmul_288x144_x128", matmul(MatMul288x144x128App),   5_000_000),
    ("matmul_432x144_x128", matmul(MatMul432x144x128App),  10_000_000),
    ("matmul_144x288_x128", matmul(MatMul144x288x128App),  10_000_000),
    ("matmul_192x192_x128", matmul(MatMul192x192x128App),  10_000_000),
    ("matmul_192x384_x128", matmul(MatMul192x384x128App),  10_000_000),
    ("matmul_240x240_x128", matmul(MatMul240x240x128App),  10_000_000),
    ("matmul_240x480_x128", matmul(MatMul240x480x128App),  10_000_000),
    ("matmul_384x192_x128", matmul(MatMul384x192x128App),  10_000_000),
    ("matmul_480x240_x128", matmul(MatMul480x240x128App),  10_000_000),
    ("matmul_576x192_x128", matmul(MatMul576x192x128App),  10_000_000),
    ("matmul_720x240_x128", matmul(MatMul720x240x128App),  20_000_000),
    ("unfold_32x32x144",    matmul(Unfold32x32x144App, has_weights=False), 5_000_000),
    ("residual_add_256x144", residual(ResidualAdd256x144App),  100_000),
    ("layernorm_128x16",    layernorm(LayerNorm128x16App),    100_000),
    ("layernorm_256x144",   layernorm(LayerNorm256x144App),  5_000_000),
    ("attn_scores_km_256x36", attn_scores(AttnScoresKM256x36App), 5_000_000),
    ("attn_v_256x36",       attn_pv(AttnV256x36App),        20_000_000),
    ("attn_v_bcast_36",     attn_pv(AttnVBcast36App),       20_000_000),
    ("qk_scores_256x36",    qk(QkScores256x36App),           5_000_000),
]

for name, build, max_cycles in APPS:
    try:
        app, state = build(name)
        run_kw = dict(max_cycles=max_cycles)
        if state is not None:
            run_kw["state"] = state
        st, cycles = app.run(**run_kw)
        print(f"\n=== {name} ===")
        print(st.stats.format_summary())
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {name} === FAILED: {e!r}")
