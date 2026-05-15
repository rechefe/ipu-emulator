"""Profile memory access patterns for depthwise_conv_stride2_small (cols=32 or 64).

The assembly is a Jinja template parameterized by cols, so each cols value
produces a distinct binary. Binaries are cached alongside the .asm file.

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_stride2_small
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_small import DepthwiseConvStride2SmallApp
from ipu_apps.convolutions_universal.profiling._utils import (
    cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

ASM = Path(__file__).resolve().parents[1] / "depthwise" / "depthwise_conv_stride2_small" / "depthwise_conv_stride2_small.asm"
CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}

# (rows, cols, channels)
CONFIGS = [
    (64,  64,  8),
    (64,  64, 16),
    (64,  64, 32),
    (64,  64, 64),
    (128, 64, 16),
    (128, 64, 32),
    (64,  32, 16),
    (64,  32, 32),
    (128, 32, 16),
]

# Cache assembled binaries per cols value so we only assemble once per cols.
_bin_cache: dict[int, Path] = {}


def _get_bin(cols: int) -> Path:
    if cols not in _bin_cache:
        bin_path = ASM.parent / f"depthwise_conv_stride2_small_cols{cols}.bin"
        if not bin_path.exists():
            from ipu_as import lark_tree  # type: ignore
            # Render the template with the cols value prepended
            template_src = ASM.read_text()
            rendered = f"{{% set cols = {cols} %}}\n" + template_src
            print(f"  Assembling stride2_small cols={cols} -> {bin_path.name} ...")
            lark_tree.assemble_to_bin_file(rendered, str(bin_path))
        _bin_cache[cols] = bin_path
    return _bin_cache[cols]


def main() -> None:
    rng = np.random.RandomState(0)
    rows_data = []

    for rows, cols, ch in CONFIGS:
        label = f"{rows}x{cols}  ch={ch}"
        input_bytes = rng.randint(0, 256, rows * cols * ch, dtype=np.uint8).tobytes()
        kernel_bytes = rng.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(input_bytes)
        krn = make_tmp_bin(kernel_bytes)
        try:
            bin_path = _get_bin(cols)
            app = DepthwiseConvStride2SmallApp(
                inst_path=bin_path,
                input_path=inp,
                kernel_path=krn,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols, channels=ch,
            )
            result = run_profile_safe(app, CR_NAMES)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows_data.append((label, result))

    print_profile_table(["inputs", "kernels", "outputs", "mask"], rows_data)


if __name__ == "__main__":
    main()
