"""Profile memory access patterns for conv_8x8 (3x3, 8x8 spatial, flexible channels).

Input data is packed into the paired-chunk layout expected by the harness.

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_conv_8x8
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ipu_apps.convolutions_universal.conv_8x8 import Conv8x8App, _build_input_data, _build_kernel_data
from ipu_apps.convolutions_universal.profiling._utils import (
    assemble_if_needed, cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

ASM = Path(__file__).resolve().parents[1] / "conv_8x8" / "conv_8x8.asm"
CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}

SPATIAL = 64  # 8x8

# (in_channels, out_channels)
CONFIGS = [
    (8,   8),
    (16,  16),
    (32,  32),
    (64,  64),
    (128, 128),
    (16,  32),
    (32,  64),
    (64,  128),
]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    rng = np.random.RandomState(0)
    rows_data = []

    for ic, oc in CONFIGS:
        label = f"8x8  ic={ic}  oc={oc}"
        raw_input = rng.randint(0, 256, ic * SPATIAL, dtype=np.uint8).tobytes()
        raw_kernel = rng.randint(0, 256, oc * ic * 9, dtype=np.uint8).tobytes()
        input_packed = _build_input_data(raw_input, ic)
        kernel_packed = _build_kernel_data(raw_kernel, ic, oc)
        inp = make_tmp_bin(input_packed)
        krn = make_tmp_bin(kernel_packed)
        try:
            app = Conv8x8App(
                inst_path=bin_path,
                input_path=inp,
                kernel_path=krn,
                output_path=None,
                in_channels=ic, out_channels=oc,
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
