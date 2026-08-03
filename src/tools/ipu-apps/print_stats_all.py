"""Print RunStats (incl. mult and lane utilization) for ALL 25 kernels on ZDlinear.

Run from src/tools/ipu-apps/ after assembling every .bin into a bin dir:

    bash assemble_all.sh /tmp/bins
    uv run python print_stats_all.py /tmp/bins

Matmuls/attn use INT8; layernorms and qk_scores use wide-vector FP32 debug.

Two different utilization numbers are reported, and the distinction is the whole
story for the L4/L5 blocks:

``mult%``   multiply-unit *time* occupancy — ``mult_active_cycles / total_cycles``.
            The emulator increments this once per bundle that issues a non-NOP
            multiply (``ipu.py:1192``); it has no notion of lanes, so it says
            nothing about how many of the 128 lanes carried useful data.
``lane%``   lane utilization — ``useful_MACs / (total_cycles * 128)``, where
            ``useful_MACs`` is declared per kernel in ``USEFUL_MACS`` below.
            This is the number that exposes a half-empty SIMD array: a kernel
            streaming 64 real tokens through a 128-lane multiply reads ~100%
            mult% but only ~50% lane%.

``useful_MACs`` counts *mathematically necessary* multiply-accumulates — the ones
the reference implementation would perform. It deliberately excludes MACs spent
on padded K-tail elements or on idle lanes, so a kernel that pads reads below
100%. Kernels whose useful-MAC count is not meaningful (data movement, or
reductions dominated by non-MAC work) are declared ``None`` and print ``--``.
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
from ipu_apps.unfold_16x16x192 import Unfold16x16x192App
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


LANES = 128

# Mathematically necessary multiply-accumulates per kernel — the MACs the
# reference implementation performs. Excludes padded-K work and idle lanes, so
# a kernel that wastes either reads below 100% lane%.
#
# Transformer matmuls: out_ch * in_ch * N_TOK (real tokens, not the padded 128).
#   The _x128 kernels put tokens in the SIMD lanes, so N_TOK < 128 is exactly the
#   half-empty-array problem: L4 kernels carry 64 real tokens, L5 only 16.
# ``None`` = useful-MAC count not meaningful (data movement, or reduction work
#   that is not MAC-shaped); prints "--".
USEFUL_MACS = {
    # --- generic FC kernels: C = A x W^T, lanes = the N (output-channel) axis ---
    # The two N=64 kernels zero-pad T[k] to 128 lanes, so they idle half the
    # array by construction and read ~44%. Legacy kernels, not on the L4/L5
    # path, but the same class of waste that AR-1 addresses.
    "matmul_128x128":       128 * 128 * 128,
    "matmul_128x64x128":    128 * 64 * 128,
    "matmul_128x64x64":     64 * 64 * 128,   # N=64 padded to 128 lanes
    "matmul_64x64x64":      64 * 64 * 64,    # N=64 padded to 128 lanes
    # --- L3: d=144, 256 tokens = 2 full groups of 128 ---
    "matmul_144x144_x128":  144 * 144 * 256,
    "matmul_288x144_x128":  288 * 144 * 256,
    "matmul_432x144_x128":  432 * 144 * 256,
    # NOTE: runs 3 full 128-wide K chunks for K=288 (=384), unlike its siblings
    # which bound the tail chunk exactly. ~33% redundant multiply work; the
    # lane% below is against real K=288, so it reads ~75%. See Phase 0 note §1.
    "matmul_144x288_x128":  144 * 288 * 256,
    # --- L4: d=192, N_TOK=64 (half-empty 128-lane array) ---
    "matmul_576x192_x128":  576 * 192 * 64,   # QKV
    "matmul_192x192_x128":  192 * 192 * 64,   # out_proj
    "matmul_384x192_x128":  384 * 192 * 64,   # FFN1
    "matmul_192x384_x128":  192 * 384 * 64,   # FFN2
    # --- L5: d=240, N_TOK=16 (7/8 of the array idle) ---
    "matmul_720x240_x128":  720 * 240 * 16,   # QKV
    "matmul_240x240_x128":  240 * 240 * 16,   # out_proj
    "matmul_480x240_x128":  480 * 240 * 16,   # FFN1
    "matmul_240x480_x128":  240 * 480 * 16,   # FFN2
    # --- L3 attention: 1 head, N=256, D=36 unless noted ---
    "qk_scores_256x36":       256 * 256 * 36,
    "attn_scores_km_256x36":  256 * 256 * 36,
    "attn_v_256x36":          4 * 256 * 256 * 36,   # 4 heads
    "attn_v_bcast_36":        4 * 256 * 256 * 36,   # 4 heads
    # --- data movement: "useful MACs" = valid tokens produced -------------
    # Unfold does 128-lane pass-through multiplies, so lane% here measures how
    # much of each 128-lane vector carried a token that survives decimation.
    # L4 fills only 64 of 128 r_acc lanes per stream (2 stripes x 32), and
    # spends 1 of every 3 bundles storing — hence ~12.5%. Recovering this is
    # the deferred packed-layout experiment (plan Section 9).
    "unfold_16x16x192":     192 * 4 * 64,    # C x streams x tokens = 49,152
    # --- not MAC-bound ---
    "unfold_32x32x144":     None,   # pure data movement
    "residual_add_256x144": None,   # elementwise; MULT is a pass-through by 1.0
    "layernorm_128x16":     None,   # reduction + rsqrt, not MAC-shaped
    "layernorm_256x144":    None,
}


def _lane_util(name, total_cycles):
    """Return (lane_util_fraction, useful_MACs) or (None, None) if undeclared."""
    macs = USEFUL_MACS.get(name)
    if macs is None or not total_cycles:
        return None, None
    return macs / (total_cycles * LANES), macs


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
    ("unfold_16x16x192",    matmul(Unfold16x16x192App, has_weights=False), 5_000_000),
    ("residual_add_256x144", residual(ResidualAdd256x144App),  100_000),
    ("layernorm_128x16",    layernorm(LayerNorm128x16App),    100_000),
    ("layernorm_256x144",   layernorm(LayerNorm256x144App),  5_000_000),
    ("attn_scores_km_256x36", attn_scores(AttnScoresKM256x36App), 5_000_000),
    ("attn_v_256x36",       attn_pv(AttnV256x36App),        20_000_000),
    ("attn_v_bcast_36",     attn_pv(AttnVBcast36App),       20_000_000),
    ("qk_scores_256x36",    qk(QkScores256x36App),           5_000_000),
]

rows = []

for name, build, max_cycles in APPS:
    try:
        app, state = build(name)
        run_kw = dict(max_cycles=max_cycles)
        if state is not None:
            run_kw["state"] = state
        st, cycles = app.run(**run_kw)
        print(f"\n=== {name} ===")
        print(st.stats.format_summary())
        lu, macs = _lane_util(name, st.stats.total_cycles)
        if lu is not None:
            print(f"Useful MACs:         {macs:>8}")
            print(f"Lane utilization:    {lu * 100:>7.1f}%  "
                  f"(useful_MACs / (cycles x {LANES}))")
        rows.append((name, st.stats.total_cycles,
                     st.stats.mult_utilization * 100, lu))
    except Exception as e:  # noqa: BLE001
        print(f"\n=== {name} === FAILED: {e!r}")
        rows.append((name, None, None, None))

# --- summary table: mult% vs lane% side by side ----------------------------
print("\n\n=== Utilization summary ===")
print(f"{'kernel':<24}{'cycles':>10}{'mult%':>8}{'lane%':>8}")
print("-" * 50)
total = 0
for name, cyc, mu, lu in rows:
    if cyc is None:
        print(f"{name:<24}{'FAILED':>10}{'--':>8}{'--':>8}")
        continue
    total += cyc
    mu_s = f"{mu:.1f}" if mu is not None else "--"
    lu_s = f"{lu * 100:.1f}" if lu is not None else "--"
    print(f"{name:<24}{cyc:>10}{mu_s:>8}{lu_s:>8}")
print("-" * 50)
print(f"{'TOTAL':<24}{total:>10}")
