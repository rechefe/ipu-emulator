"""Run profiling sweeps for all MobileViT-S layer configs and save results.

Covers:
  - conv_first_layer       (256x256x3 -> 128x128x16, stride 2, fixed)
  - conv_universal         (3x3, flexible spatial/channels)
  - conv_8x8               (3x3, 8x8 spatial tiles)
  - depthwise_conv_universal  (3x3 depthwise, no stride)
  - depthwise_conv_stride2    (3x3 depthwise, stride 2, cols=128 fixed)
  - depthwise_8x8          (3x3 depthwise, 8x8 spatial tiles)
  - pointwise_conv_universal  (1x1, flexible spatial/channels)
  - pointwise_8x8          (1x1, 8x8 spatial tiles)

Results are written to profiling/results/<app_name>.csv

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \\
        python -m ipu_apps.convolutions_universal.profiling.run_mobilevit_sweep
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (allows running without installing packages)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[6]
for _pkg in ("ipu-emu-py/src", "ipu-common/src", "ipu-apps/src", "ipu-as-py/src"):
    _p = str(_ROOT / "src" / "tools" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ipu_apps.convolutions_universal.profiling._utils import (  # noqa: E402
    assemble_if_needed, cleanup, make_tmp_bin, run_profile_safe,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RNG = np.random.RandomState(0)

# ---------------------------------------------------------------------------
# MobileViT-S layer configs
# ---------------------------------------------------------------------------

# Regular 3x3 convolutions: (rows, cols, in_ch, out_ch)
# 256x256x3->128x128x16 handled by conv_first_layer; rest go to conv_universal or conv_8x8
CONV3X3_CONFIGS = [
    # label,                rows, cols, ic,  oc
    ("32x32_ic96_oc96",      32,   32,  96,  96),
    ("32x32_ic192_oc96",     32,   32, 192,  96),
    ("16x16_ic128_oc128",    16,   16, 128, 128),
    ("16x16_ic256_oc128",    16,   16, 256, 128),
    ("8x8_ic160_oc160",       8,    8, 160, 160),
    ("8x8_ic320_oc160",       8,    8, 320, 160),
]

# Depthwise 3x3 (no stride): (rows, cols, ch)
DW_CONFIGS = [
    ("128x128_ch64",   128, 128,  64),
    ("64x64_ch256_a",   64,  64, 256),
    ("64x64_ch256_b",   64,  64, 256),
]

# Depthwise 3x3 stride-2 (cols=128 fixed): (rows, ch)
DW_STRIDE2_CONFIGS = [
    ("128x128_ch128", 128, 128),
    ("64x64_ch256",    64, 256),
    ("32x32_ch384",    32, 384),
    ("16x16_ch512",    16, 512),
]

# Depthwise 8x8 (spatial tile 8x8): (ch,)
DW_8X8_CONFIGS = [
    ("8x8_ch64",   64),
    ("8x8_ch128", 128),
    ("8x8_ch256", 256),
    ("8x8_ch384", 384),
    ("8x8_ch512", 512),
]

# Pointwise 1x1 (non-8x8 spatial): (rows, cols, ic, oc)
PW_CONFIGS = [
    ("128x128_ic16_oc64",   128, 128,  16,  64),
    ("128x128_ic64_oc32",   128, 128,  64,  32),
    ("128x128_ic32_oc128",  128, 128,  32, 128),
    ("64x64_ic128_oc64",     64,  64, 128,  64),
    ("64x64_ic64_oc256_a",   64,  64,  64, 256),
    ("64x64_ic256_oc64_a",   64,  64, 256,  64),
    ("64x64_ic64_oc256_b",   64,  64,  64, 256),
    ("64x64_ic256_oc64_b",   64,  64, 256,  64),
    ("64x64_ic64_oc256_c",   64,  64,  64, 256),
    ("32x32_ic256_oc96",     32,  32, 256,  96),
    ("32x32_ic96_oc144",     32,  32,  96, 144),
    ("32x32_ic144_oc96",     32,  32, 144,  96),
    ("32x32_ic96_oc384",     32,  32,  96, 384),
    ("16x16_ic384_oc128",    16,  16, 384, 128),
    ("16x16_ic128_oc192",    16,  16, 128, 192),
    ("16x16_ic192_oc128",    16,  16, 192, 128),
    ("16x16_ic128_oc512",    16,  16, 128, 512),
]

# Pointwise 8x8 (spatial tile 8x8): (ic, oc)
PW_8X8_CONFIGS = [
    ("8x8_ic512_oc160",  512, 160),
    ("8x8_ic160_oc240",  160, 240),
    ("8x8_ic240_oc160",  240, 160),
    ("8x8_ic160_oc640",  160, 640),
]


# ---------------------------------------------------------------------------
# CSV writer helpers
# ---------------------------------------------------------------------------

def _write_csv(
    filename: str,
    col_headers: list[str],
    rows: list[tuple[str, dict[str, int] | str]],
) -> None:
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config"] + col_headers)
        for label, result in rows:
            if isinstance(result, str):
                writer.writerow([label] + ["ERROR: " + result] + [""] * (len(col_headers) - 1))
            else:
                writer.writerow([label] + [result.get(h, "") for h in col_headers])
    print(f"  -> saved {path.relative_to(RESULTS_DIR.parent.parent.parent.parent.parent.parent.parent)}")


def _print_table(
    col_headers: list[str],
    rows: list[tuple[str, dict[str, int] | str]],
    col_width: int = 9,
) -> None:
    label_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Config':<{label_w}}" + "".join(f"{h:>{col_width}}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for label, result in rows:
        if isinstance(result, str):
            print(f"{label:<{label_w}} ERROR: {result}")
        else:
            vals = "".join(f"{result.get(h, '-'):>{col_width}}" for h in col_headers)
            print(f"{label:<{label_w}}{vals}")
    print()


# ---------------------------------------------------------------------------
# Per-app sweep functions
# ---------------------------------------------------------------------------

def run_conv_first_layer() -> None:
    print("=== conv_first_layer ===")
    from ipu_apps.convolutions_universal.conv_first_layer import (
        ConvFirstLayerApp, IN_ROWS, IN_COLS, IN_CHANNELS, OUT_CHANNELS,
    )
    ASM = Path(__file__).resolve().parents[1] / "conv_first_layer" / "conv_first_layer.asm"
    CR_NAMES = {0: "input_ch0", 1: "input_ch1", 2: "input_ch2",
                3: "kernels", 4: "mask", 5: "outputs", 6: "temp"}
    COLS = ["input_ch0", "input_ch1", "input_ch2", "kernels", "mask", "outputs", "temp"]

    bin_path = assemble_if_needed(ASM)
    inp_b = RNG.randint(0, 256, IN_ROWS * IN_COLS * IN_CHANNELS, dtype=np.uint8).tobytes()
    krn_b = RNG.randint(0, 256, OUT_CHANNELS * IN_CHANNELS * 9, dtype=np.uint8).tobytes()
    inp = make_tmp_bin(inp_b)
    krn = make_tmp_bin(krn_b)
    try:
        app = ConvFirstLayerApp(inst_path=bin_path, input_path=inp, kernel_path=krn, output_path=None)
        result = run_profile_safe(app, CR_NAMES, max_cycles=50_000_000)
    finally:
        cleanup(inp, krn)

    label = f"{IN_ROWS}x{IN_COLS}x{IN_CHANNELS}_oc{OUT_CHANNELS}_stride2"
    rows = [(label, result)]
    _print_table(COLS, rows)
    _write_csv("conv_first_layer.csv", COLS, rows)


def run_conv_universal() -> None:
    print("=== conv_universal ===")
    from ipu_apps.convolutions_universal.conv_universal import ConvUniversalApp
    BIN = Path(__file__).resolve().parents[1] / "conv_universal" / "conv_universal.bin"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]

    rows = []
    for label, r, c, ic, oc in CONV3X3_CONFIGS:
        inp_b = RNG.randint(0, 256, r * c * ic, dtype=np.uint8).tobytes()
        krn_b = RNG.randint(0, 256, oc * ic * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(inp_b)
        krn = make_tmp_bin(krn_b)
        try:
            app = ConvUniversalApp(
                inst_path=BIN, input_path=inp, kernel_path=krn, output_path=None,
                dtype="INT8", rows=r, cols=c, in_channels=ic, out_channels=oc,
            )
            result = run_profile_safe(app, CR_NAMES, max_cycles=50_000_000)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("conv_universal.csv", COLS, rows)


def run_conv_8x8() -> None:
    print("=== conv_8x8 ===")
    from ipu_apps.convolutions_universal.conv_8x8 import Conv8x8App, _build_input_data, _build_kernel_data
    ASM = Path(__file__).resolve().parents[1] / "conv_8x8" / "conv_8x8.asm"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]
    SPATIAL = 64  # 8x8 tile = 64 elements

    bin_path = assemble_if_needed(ASM)
    # MobileViT 8x8 spatial configs: (label, ic, oc)
    configs_8x8 = [
        ("8x8_ic160_oc160", 160, 160),
        ("8x8_ic320_oc160", 320, 160),
    ]

    rows = []
    for label, ic, oc in configs_8x8:
        raw_inp = RNG.randint(0, 256, ic * SPATIAL, dtype=np.uint8).tobytes()
        raw_krn = RNG.randint(0, 256, oc * ic * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(_build_input_data(raw_inp, ic))
        krn = make_tmp_bin(_build_kernel_data(raw_krn, ic, oc))
        try:
            app = Conv8x8App(
                inst_path=bin_path, input_path=inp, kernel_path=krn, output_path=None,
                in_channels=ic, out_channels=oc,
            )
            result = run_profile_safe(app, CR_NAMES)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("conv_8x8.csv", COLS, rows)


def run_depthwise_universal() -> None:
    print("=== depthwise_conv_universal ===")
    from ipu_apps.convolutions_universal.depthwise_conv_universal import DepthwiseConvUniversalApp
    BIN = Path(__file__).resolve().parents[1] / "depthwise_conv_universal" / "depthwise_conv_universal.bin"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]

    rows = []
    for label, r, c, ch in DW_CONFIGS:
        inp_b = RNG.randint(0, 256, r * c * ch, dtype=np.uint8).tobytes()
        krn_b = RNG.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(inp_b)
        krn = make_tmp_bin(krn_b)
        try:
            app = DepthwiseConvUniversalApp(
                inst_path=BIN, input_path=inp, kernel_path=krn, output_path=None,
                dtype="INT8", rows=r, cols=c, channels=ch,
            )
            result = run_profile_safe(app, CR_NAMES, max_cycles=50_000_000)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("depthwise_conv_universal.csv", COLS, rows)


def run_depthwise_stride2() -> None:
    print("=== depthwise_conv_stride2 ===")
    from ipu_apps.convolutions_universal.depthwise_conv_stride2 import DepthwiseConvStride2App
    ASM = Path(__file__).resolve().parents[1] / "depthwise_conv_stride2" / "depthwise_conv_stride2.asm"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]
    COLS_FIXED = 128

    bin_path = assemble_if_needed(ASM)
    rows = []
    for label, r, ch in DW_STRIDE2_CONFIGS:
        inp_b = RNG.randint(0, 256, r * COLS_FIXED * ch, dtype=np.uint8).tobytes()
        krn_b = RNG.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(inp_b)
        krn = make_tmp_bin(krn_b)
        try:
            app = DepthwiseConvStride2App(
                inst_path=bin_path, input_path=inp, kernel_path=krn, output_path=None,
                dtype="INT8", rows=r, cols=COLS_FIXED, channels=ch,
            )
            result = run_profile_safe(app, CR_NAMES, max_cycles=50_000_000)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("depthwise_conv_stride2.csv", COLS, rows)


def run_depthwise_8x8() -> None:
    print("=== depthwise_8x8 ===")
    from ipu_apps.convolutions_universal.depthwise_8x8 import Depthwise8x8App, _build_input_data, _build_kernel_data
    ASM = Path(__file__).resolve().parents[1] / "depthwise_8x8" / "depthwise_8x8.asm"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]
    SPATIAL = 64

    bin_path = assemble_if_needed(ASM)
    rows = []
    for label, ch in DW_8X8_CONFIGS:
        raw_inp = RNG.randint(0, 256, ch * SPATIAL, dtype=np.uint8).tobytes()
        raw_krn = RNG.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(_build_input_data(raw_inp, ch))
        krn = make_tmp_bin(_build_kernel_data(raw_krn, ch))
        try:
            app = Depthwise8x8App(
                inst_path=bin_path, input_path=inp, kernel_path=krn, output_path=None,
                num_channels=ch,
            )
            result = run_profile_safe(app, CR_NAMES)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("depthwise_8x8.csv", COLS, rows)


def run_pointwise_universal() -> None:
    print("=== pointwise_conv_universal ===")
    from ipu_apps.convolutions_universal.pointwise_conv_universal import PointwiseConvUniversalApp
    BIN = Path(__file__).resolve().parents[1] / "pointwise_conv_universal" / "pointwise_conv_universal.bin"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "mask", 3: "outputs"}
    COLS = ["inputs", "kernels", "mask", "outputs"]

    rows = []
    for label, r, c, ic, oc in PW_CONFIGS:
        inp_b = RNG.randint(0, 256, r * c * ic, dtype=np.uint8).tobytes()
        krn_b = RNG.randint(0, 256, oc * ic, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(inp_b)
        krn = make_tmp_bin(krn_b)
        try:
            app = PointwiseConvUniversalApp(
                inst_path=BIN, input_path=inp, kernel_path=krn, output_path=None,
                dtype="INT8", rows=r, cols=c, in_channels=ic, out_channels=oc,
            )
            result = run_profile_safe(app, CR_NAMES, max_cycles=50_000_000)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("pointwise_conv_universal.csv", COLS, rows)


def run_pointwise_8x8() -> None:
    print("=== pointwise_8x8 ===")
    from ipu_apps.convolutions_universal.pointwise_8x8 import Pointwise8x8App, _build_input_data, _build_kernel_data
    ASM = Path(__file__).resolve().parents[1] / "pointwise_8x8" / "pointwise_8x8.asm"
    CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}
    COLS = ["inputs", "kernels", "outputs", "mask"]
    SPATIAL = 64

    bin_path = assemble_if_needed(ASM)
    rows = []
    for label, ic, oc in PW_8X8_CONFIGS:
        raw_inp = RNG.randint(0, 256, ic * SPATIAL, dtype=np.uint8).tobytes()
        raw_krn = RNG.randint(0, 256, oc * ic, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(_build_input_data(raw_inp, ic))
        krn = make_tmp_bin(_build_kernel_data(raw_krn, ic, oc))
        try:
            app = Pointwise8x8App(
                inst_path=bin_path, input_path=inp, kernel_path=krn, output_path=None,
                in_channels=ic, out_channels=oc,
            )
            result = run_profile_safe(app, CR_NAMES)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows.append((label, result))

    _print_table(COLS, rows)
    _write_csv("pointwise_8x8.csv", COLS, rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Saving results to: {RESULTS_DIR}\n")
    run_conv_first_layer()
    run_conv_universal()
    run_conv_8x8()
    run_depthwise_universal()
    run_depthwise_stride2()
    run_depthwise_8x8()
    run_pointwise_universal()
    run_pointwise_8x8()
    print("Done.")


if __name__ == "__main__":
    main()
