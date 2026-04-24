"""Benchmark: conv_8x8 (3x3 standard conv, 8x8 spatial) on MobileViT-S configurations.

MobileViT-S 3x3 conv layers at 8x8 spatial (Stage-3):
  Stage 3 Local:  8x8 ic=160 oc=160
  Stage 3 Fusion: 8x8 ic=320 oc=160
  Plus two larger-channel configs from Stage-2 at 8x8.

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_conv_8x8
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.conv_8x8 import (
    Conv8x8App,
    OUTPUT_BASE_ADDR,
    _build_input_data,
    _build_kernel_data,
)
from ipu_apps.convolutions_universal.profiling._utils import assemble_if_needed
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    ref_conv3x3,
    read_acc_paired_8x8,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

ASM = (
    Path(__file__).resolve().parents[1]
    / "conv_8x8"
    / "conv_8x8.asm"
)

ROWS, COLS, SPATIAL = 8, 8, 64

# (name, in_ch, out_ch)  -- MobileViT-S 8x8 3x3 standard conv layers
CONFIGS = [
    ("8x8 ic=160 oc=160  Stage3-loc", 160, 160),
    ("8x8 ic=320 oc=160  Stage3-fus", 320, 160),
    ("8x8 ic=128 oc=128  Stage2-loc", 128, 128),
    ("8x8 ic=256 oc=128  Stage2-fus", 256, 128),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, in_ch, out_ch in CONFIGS:
            input_chw = gen_int8((in_ch, ROWS, COLS))
            kernel_oihw = gen_int8((out_ch, in_ch, 3, 3))

            input_raw = input_chw.reshape(in_ch, SPATIAL).view(np.uint8).tobytes()
            kernel_raw = kernel_oihw.view(np.uint8).tobytes()
            input_packed = _build_input_data(input_raw, in_ch)
            kernel_packed = _build_kernel_data(kernel_raw, in_ch, out_ch)

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_packed)

            app = Conv8x8App(
                inst_path=bin_path,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                in_channels=in_ch,
                out_channels=out_ch,
            )
            state, cycles = app.run(max_cycles=100_000_000)

            actual = read_acc_paired_8x8(state, OUTPUT_BASE_ADDR, out_ch, SPATIAL)
            # actual: (out_ch, SPATIAL)

            expected = ref_conv3x3(input_chw, kernel_oihw).numpy()
            expected_flat = expected.reshape(out_ch, SPATIAL)
            correct = check_correct(actual, expected_flat, name)

            macs = SPATIAL * in_ch * out_ch * 9
            print_table_row(name, cycles, macs, correct)

    print_table_footer("conv_8x8")


if __name__ == "__main__":
    main()
