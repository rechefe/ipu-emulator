"""Benchmark: pointwise_8x8 on MobileViT-S configurations (8x8 spatial).

MobileViT-S pointwise layers at 8x8 spatial (Stage-3 / final):
  160->640  (Final expansion)
  512->160  (Project-5)
  160->640  repeated for symmetry is above; use 4 distinct configs below.

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_pointwise_8x8
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.pointwise_8x8 import (
    Pointwise8x8App,
    OUTPUT_BASE_ADDR,
    _build_input_data,
    _build_kernel_data,
)
from ipu_apps.convolutions_universal.profiling._utils import assemble_if_needed
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    ref_pointwise,
    read_acc_paired_8x8,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

ASM = (
    Path(__file__).resolve().parents[1]
    / "pointwise_8x8"
    / "pointwise_8x8.asm"
)

ROWS, COLS, SPATIAL = 8, 8, 64

# (name, in_ch, out_ch)  -- MobileViT-S 8x8 pointwise layers
CONFIGS = [
    ("8x8 160->640  Final-exp",  160, 640),
    ("8x8 512->160  Project-5",  512, 160),
    ("8x8 160->160  Stage3-loc", 160, 160),
    ("8x8  96->384  Expand-4",    96, 384),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, in_ch, out_ch in CONFIGS:
            input_chw = gen_int8((in_ch, ROWS, COLS))
            kernel_oihw = gen_int8((out_ch, in_ch, 1, 1))

            input_raw = input_chw.reshape(in_ch, SPATIAL).view(np.uint8).tobytes()
            kernel_raw = kernel_oihw.reshape(out_ch, in_ch).view(np.uint8).tobytes()
            input_packed = _build_input_data(input_raw, in_ch)
            kernel_packed = _build_kernel_data(kernel_raw, in_ch, out_ch)

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_packed)

            app = Pointwise8x8App(
                inst_path=bin_path,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                in_channels=in_ch,
                out_channels=out_ch,
            )
            state, cycles = app.run(max_cycles=50_000_000)

            actual = read_acc_paired_8x8(state, OUTPUT_BASE_ADDR, out_ch, SPATIAL)
            # actual: (out_ch, SPATIAL)

            expected = ref_pointwise(input_chw, kernel_oihw).numpy()
            # expected: (out_ch, ROWS, COLS) -> flatten spatial
            expected_flat = expected.reshape(out_ch, SPATIAL)
            correct = check_correct(actual, expected_flat, name)

            macs = SPATIAL * in_ch * out_ch
            print_table_row(name, cycles, macs, correct)

    print_table_footer("pointwise_8x8")


if __name__ == "__main__":
    main()
