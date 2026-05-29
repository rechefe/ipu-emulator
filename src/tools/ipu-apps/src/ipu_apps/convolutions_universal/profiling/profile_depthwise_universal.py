"""Profile memory access patterns for depthwise_conv_universal (3x3 depthwise).

Sweeps spatial sizes and channel counts, printing peak lookahead rows
per memory region (inputs, kernels, outputs, mask).

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_universal
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import DepthwiseConvUniversalApp
from ipu_apps.convolutions_universal.profiling._utils import (
    assemble_if_needed, cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

ASM = Path(__file__).resolve().parents[1] / "depthwise" / "depthwise_conv_universal" / "depthwise_conv_universal.asm"
CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}

# (rows, cols, channels)
CONFIGS = [
    (64,  64,   8),
    (64,  64,  16),
    (64,  64,  32),
    (64,  64,  64),
    (64,  64, 128),
    (64,  64, 256),
    (16,  16,  16),
    (32,  32,  16),
    (128, 128, 16),
    (32,  32,  32),
    (32,  32,  64),
    (128, 128, 32),
    (128, 128, 64),
    (16,  16,  64),
    (16,  16, 128),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    rng = np.random.RandomState(0)
    rows_data = []

    for rows, cols, ch in CONFIGS:
        label = f"{rows}x{cols}  ch={ch}"
        input_bytes = rng.randint(0, 256, rows * cols * ch, dtype=np.uint8).tobytes()
        kernel_bytes = rng.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(input_bytes)
        krn = make_tmp_bin(kernel_bytes)
        try:
            app = DepthwiseConvUniversalApp(
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
