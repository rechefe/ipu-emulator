"""Benchmark: depthwise_conv_stride2 on MobileViT-S downsampling configurations.

MobileViT-S stride-2 depthwise layers (cols=128, from REQUIRED_CONVOLUTIONS.md):
  Downsample 1: 128x128 ch=128  ->  64x64 ch=128
  Downsample 2:  64x128 ch=256  ->  32x64 ch=256  (note: rows=64, cols=128)
  Downsample 3:  32x128 ch=384  ->  16x64 ch=384
  Downsample 4:  16x128 ch=512  ->   8x64 ch=512

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_depthwise_stride2
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise_conv_stride2 import (
    DepthwiseConvStride2App,
    OUTPUT_BASE_ADDR,
)
from ipu_apps.convolutions_universal.profiling._utils import assemble_if_needed
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    pack_input_multichannel,
    ref_depthwise_stride2_quantized,
    read_int8_multichannel_output,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

ASM = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_stride2"
    / "depthwise_conv_stride2.asm"
)

COLS = 128  # fixed for this app

# (name, rows, channels)
CONFIGS = [
    ("DW-S2 128x128 ch=128  Down1", 128, 128),
    ("DW-S2  64x128 ch=256  Down2",  64, 256),
    ("DW-S2  32x128 ch=384  Down3",  32, 384),
    ("DW-S2  16x128 ch=512  Down4",  16, 512),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, ch in CONFIGS:
            input_chw = gen_int8((ch, rows, COLS))
            kernel_c33 = gen_int8((ch, 3, 3))

            input_packed = pack_input_multichannel(input_chw, rows, COLS, ch)
            kernel_bytes = kernel_c33.view(np.uint8).tobytes()

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_bytes)

            app = DepthwiseConvStride2App(
                inst_path=bin_path,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=COLS, channels=ch,
            )
            state, cycles = app.run(max_cycles=100_000_000)

            out_rows = rows // 2
            out_cols = COLS // 2
            rows_per_out_chunk = 128 // out_cols  # =2
            num_out_chunks = out_rows // rows_per_out_chunk
            actual = read_int8_multichannel_output(
                state, OUTPUT_BASE_ADDR, num_out_chunks, ch, rows_per_out_chunk, out_cols
            )  # (ch, out_rows, out_cols)

            expected = ref_depthwise_stride2_quantized(input_chw, kernel_c33)
            correct = check_correct(actual, expected, name)

            macs = out_rows * out_cols * ch * 9
            print_table_row(name, cycles, macs, correct)

    print_table_footer("depthwise_conv_stride2")


if __name__ == "__main__":
    main()
