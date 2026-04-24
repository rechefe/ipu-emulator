"""Benchmark: depthwise_conv_universal on MobileViT-S configurations.

MobileViT-S depthwise stride-1 layers (from REQUIRED_CONVOLUTIONS.md):
  Early Stage:   128x128 ch=64
  Mid Stage x2:   64x64  ch=256
  Stage 1 local:  32x32  ch=96   (3x3 depthwise in inverted residual)
  Stage 2 local:  16x16  ch=128

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_depthwise_universal
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
    OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    pack_input_multichannel,
    ref_depthwise,
    read_universal_output,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

BIN = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_universal"
    / "depthwise_conv_universal.bin"
)

# (name, rows, cols, channels)
CONFIGS = [
    ("DW  128x128 ch=64   Early",  128, 128,  64),
    ("DW   64x64  ch=256  Mid",     64,  64, 256),
    ("DW   32x32  ch=96   Stg1",    32,  32,  96),
    ("DW   16x16  ch=128  Stg2",    16,  16, 128),
]


def main() -> None:
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, cols, ch in CONFIGS:
            input_chw = gen_int8((ch, rows, cols))
            kernel_c33 = gen_int8((ch, 3, 3))

            input_packed = pack_input_multichannel(input_chw, rows, cols, ch)
            kernel_bytes = kernel_c33.view(np.uint8).tobytes()

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_bytes)

            app = DepthwiseConvUniversalApp(
                inst_path=BIN,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols, channels=ch,
            )
            state, cycles = app.run(max_cycles=50_000_000)

            rows_per_chunk = 128 // cols
            num_chunks = (rows * cols) // 128
            actual = read_universal_output(
                state, OUTPUT_BASE_ADDR, num_chunks, ch, rows_per_chunk, cols
            )  # (ch, rows, cols)

            expected = ref_depthwise(input_chw, kernel_c33).numpy()
            correct = check_correct(actual, expected, name)

            macs = rows * cols * ch * 9  # 3x3 kernel
            print_table_row(name, cycles, macs, correct)

    print_table_footer("depthwise_conv_universal")


if __name__ == "__main__":
    main()
