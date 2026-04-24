"""Benchmark: depthwise_conv_stride2_small on MobileViT-S downsampling configurations.

Handles cols=32 or cols=64 (Jinja-templated binary, one per cols value).

MobileViT-S stride-2 depthwise layers at smaller spatial widths:
  64x64 ch=64   -> 32x32 ch=64   (Early stage down)
  128x64 ch=256 -> 64x32 ch=256  (Mid stage down, viewed as rows=128 cols=64)
  64x32 ch=384  -> 32x16 ch=384
  128x32 ch=512 -> 64x16 ch=512

Run from repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.benchmarks.bench_depthwise_stride2_small
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise_conv_stride2_small import (
    DepthwiseConvStride2SmallApp,
    OUTPUT_BASE_ADDR,
)
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

ASM_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "depthwise_conv_stride2_small"
    / "depthwise_conv_stride2_small.asm"
)

# (name, rows, cols, channels)
CONFIGS = [
    ("DW-S2s  64x64  ch=64   EarlyD",   64,  64,  64),
    ("DW-S2s 128x64  ch=256  MidD",     128,  64, 256),
    ("DW-S2s  64x32  ch=384  Stg3D",     64,  32, 384),
    ("DW-S2s 128x32  ch=512  Stg4D",    128,  32, 512),
]

# Cache assembled binaries per cols value
_bin_cache: dict[int, Path] = {}


def _get_bin(cols: int) -> Path:
    if cols not in _bin_cache:
        bin_path = ASM_TEMPLATE.parent / f"depthwise_conv_stride2_small_cols{cols}.bin"
        if not bin_path.exists():
            from ipu_as import lark_tree
            template_src = ASM_TEMPLATE.read_text()
            rendered = f"{{% set cols = {cols} %}}\n" + template_src
            print(f"  Assembling stride2_small cols={cols} -> {bin_path.name} ...")
            lark_tree.assemble_to_bin_file(rendered, str(bin_path))
        _bin_cache[cols] = bin_path
    return _bin_cache[cols]


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

            bin_path = _get_bin(cols)
            app = DepthwiseConvStride2SmallApp(
                inst_path=bin_path,
                input_path=inp_f,
                kernel_path=krn_f,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols, channels=ch,
            )
            state, cycles = app.run(max_cycles=100_000_000)

            out_rows = rows // 2
            out_cols = cols // 2
            num_chunks = (rows * cols) // 128
            num_groups = num_chunks // 4
            rows_per_out_chunk = 128 // out_cols
            actual = read_int8_multichannel_output(
                state, OUTPUT_BASE_ADDR, num_groups, ch, rows_per_out_chunk, out_cols
            )  # (ch, out_rows, out_cols)

            expected = ref_depthwise_stride2_quantized(input_chw, kernel_c33)
            correct = check_correct(actual, expected, name)

            macs = out_rows * out_cols * ch * 9
            print_table_row(name, cycles, macs, correct)

    print_table_footer("depthwise_conv_stride2_small")


if __name__ == "__main__":
    main()
