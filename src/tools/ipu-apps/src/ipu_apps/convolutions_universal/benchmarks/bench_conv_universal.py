"""Benchmark: conv_universal (3x3 stride-1) on MobileViT-S configurations.

MobileViT-S standard 3x3 stride-1 layers (from REQUIRED_CONVOLUTIONS.md):
  Stage 1 Local:  32x32 ic=96  oc=96
  Stage 1 Fusion: 32x32 ic=192 oc=96
  Stage 2 Local:  16x16 ic=128 oc=128
  Stage 2 Fusion: 16x16 ic=256 oc=128

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_conv_universal
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.conv_universal import (
    ConvUniversalApp,
    OUTPUT_BASE_ADDR,
)
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    pack_input_multichannel,
    ref_conv3x3,
    read_universal_output,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

BIN = (
    Path(__file__).resolve().parents[1]
    / "conv_universal"
    / "conv_universal.bin"
)

# (name, rows, cols, in_ch, out_ch)
CONFIGS = [
    ("Conv  32x32 ic=96  oc=96   Stg1L",  32, 32,  96,  96),
    ("Conv  32x32 ic=192 oc=96   Stg1F",  32, 32, 192,  96),
    ("Conv  16x16 ic=128 oc=128  Stg2L",  16, 16, 128, 128),
    ("Conv  16x16 ic=256 oc=128  Stg2F",  16, 16, 256, 128),
]


def main() -> None:
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, cols, in_ch, out_ch in CONFIGS:
            input_chw = gen_int8((in_ch, rows, cols))
            kernel_oihw = gen_int8((out_ch, in_ch, 3, 3))

            input_packed = pack_input_multichannel(input_chw, rows, cols, in_ch)
            kernel_bytes = kernel_oihw.view(np.uint8).tobytes()

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_bytes)

            app = ConvUniversalApp(
                inst_path=BIN,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols,
                in_channels=in_ch, out_channels=out_ch,
            )
            state, cycles = app.run(max_cycles=200_000_000)

            rows_per_chunk = 128 // cols
            num_chunks = (rows * cols) // 128
            actual = read_universal_output(
                state, OUTPUT_BASE_ADDR, num_chunks, out_ch, rows_per_chunk, cols
            )  # (out_ch, rows, cols)

            expected = ref_conv3x3(input_chw, kernel_oihw).numpy()
            correct = check_correct(actual, expected, name)

            macs = rows * cols * in_ch * out_ch * 9
            print_table_row(name, cycles, macs, correct)

    print_table_footer("conv_universal")


if __name__ == "__main__":
    main()
