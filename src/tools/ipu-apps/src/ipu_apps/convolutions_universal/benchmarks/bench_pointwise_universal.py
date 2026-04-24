"""Benchmark: pointwise_conv_universal on MobileViT-S configurations.

MobileViT-S pointwise layers (from REQUIRED_CONVOLUTIONS.md):
  Expansions:  16->64 @ 128x128,  64->256 @ 64x64,  96->384 @ 32x32,  128->512 @ 16x16
  Projections: 64->32 @ 128x128, 128->64 @ 64x64,  256->96 @ 32x32,  384->128 @ 16x16

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_pointwise_universal
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.pointwise_conv_universal import (
    PointwiseConvUniversalApp,
    OUTPUT_BASE_ADDR,
)
from ipu_apps.convolutions_universal.benchmarks._utils import (
    gen_int8,
    pack_input_multichannel,
    ref_pointwise,
    read_universal_output,
    check_correct,
    print_table_header,
    print_table_row,
    print_table_footer,
)

BIN = (
    Path(__file__).resolve().parents[1]
    / "pointwise_conv_universal"
    / "pointwise_conv_universal.bin"
)

# (name, rows, cols, in_ch, out_ch)
CONFIGS = [
    ("Expand-1  128x128 16->64",    128, 128,  16,  64),
    ("Expand-3   64x64  64->256",    64,  64,  64, 256),
    ("Expand-4   32x32  96->384",    32,  32,  96, 384),
    ("Expand-5   16x16 128->512",    16,  16, 128, 512),
    ("Project-1 128x128 64->32",    128, 128,  64,  32),
    ("Project-2  64x64 128->64",     64,  64, 128,  64),
    ("Project-3  32x32 256->96",     32,  32, 256,  96),
    ("Project-4  16x16 384->128",    16,  16, 384, 128),
]


def main() -> None:
    print_table_header()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, rows, cols, in_ch, out_ch in CONFIGS:
            input_chw = gen_int8((in_ch, rows, cols))
            kernel_raw = gen_int8((out_ch, in_ch, 1, 1))

            input_packed = pack_input_multichannel(input_chw, rows, cols, in_ch)
            kernel_bytes = kernel_raw.reshape(out_ch, in_ch).view(np.uint8).tobytes()

            inp_f = tmp / f"inp_{name[:8]}.bin"
            krn_f = tmp / f"krn_{name[:8]}.bin"
            inp_f.write_bytes(input_packed)
            krn_f.write_bytes(kernel_bytes)

            app = PointwiseConvUniversalApp(
                inst_path=BIN,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols,
                in_channels=in_ch, out_channels=out_ch,
            )
            state, cycles = app.run(max_cycles=50_000_000)

            # Read output: out_ch chunks of 512B each (one per channel per row-group)
            rows_per_chunk = 128 // cols
            num_chunks = (rows * cols) // 128
            actual = read_universal_output(
                state, OUTPUT_BASE_ADDR, num_chunks, out_ch, rows_per_chunk, cols
            )  # (out_ch, rows, cols)

            expected = ref_pointwise(input_chw, kernel_raw).numpy()  # (out_ch, rows, cols)
            correct = check_correct(actual, expected, name)

            macs = rows * cols * in_ch * out_ch
            print_table_row(name, cycles, macs, correct)

    print_table_footer("pointwise_conv_universal")


if __name__ == "__main__":
    main()
