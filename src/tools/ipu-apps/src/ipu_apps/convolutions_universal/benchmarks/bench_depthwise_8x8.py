"""Benchmark: depthwise_8x8 on MobileViT-S configurations (8x8 spatial).

MobileViT-S depthwise layers at 8x8 spatial (Stage-3):
  Stage 3 local:  8x8 ch=160  (3x3 depthwise in inverted residual)
  Stage 3 fusion: 8x8 ch=320  (after concat)
  Plus two larger-channel configs from the Stage-3 expansion path.

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_depthwise_8x8
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise_8x8 import (
    Depthwise8x8App,
    OUTPUT_BASE_ADDR,
    _build_input_data,
    _build_kernel_data,
)
from ipu_apps.convolutions_universal.profiling._utils import assemble_if_needed
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    ref_depthwise,
    read_acc_paired_8x8,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

ASM = (
    Path(__file__).resolve().parents[1]
    / "depthwise_8x8"
    / "depthwise_8x8.asm"
)

ROWS, COLS, SPATIAL = 8, 8, 64

# (name, num_channels)  -- MobileViT-S 8x8 depthwise layers
CONFIGS = [
    ("DW 8x8 ch=160  Stage3-loc", 160),
    ("DW 8x8 ch=320  Stage3-fus", 320),
    ("DW 8x8 ch=128  Stage2-loc", 128),
    ("DW 8x8 ch=256  Stage2-fus", 256),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, ch in CONFIGS:
            input_chw = gen_int8((ch, ROWS, COLS))
            kernel_c33 = gen_int8((ch, 3, 3))

            input_raw = input_chw.reshape(ch, SPATIAL).view(np.uint8).tobytes()
            kernel_raw = kernel_c33.view(np.uint8).tobytes()
            input_packed = _build_input_data(input_raw, ch)
            kernel_packed = _build_kernel_data(kernel_raw, ch)

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_packed)

            app = Depthwise8x8App(
                inst_path=bin_path,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                num_channels=ch,
            )
            state, cycles = app.run(max_cycles=50_000_000)

            actual = read_acc_paired_8x8(state, OUTPUT_BASE_ADDR, ch, SPATIAL)
            # actual: (ch, SPATIAL)

            expected = ref_depthwise(input_chw, kernel_c33).numpy()
            expected_flat = expected.reshape(ch, SPATIAL)
            correct = check_correct(actual, expected_flat, name)

            macs = SPATIAL * ch * 9
            print_table_row(name, cycles, macs, correct)

    print_table_footer("depthwise_8x8")


if __name__ == "__main__":
    main()
