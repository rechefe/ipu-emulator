"""Profile memory access patterns for depthwise_8x8 (3x3 depthwise, 8x8 spatial).

Input data is packed into the paired-chunk layout expected by the harness.

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_depthwise_8x8
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ipu_apps.convolutions_universal.depthwise.depthwise_8x8 import Depthwise8x8App, _build_input_data, _build_kernel_data
from ipu_apps.convolutions_universal.profiling._utils import (
    assemble_if_needed, cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

ASM = Path(__file__).resolve().parents[1] / "depthwise" / "depthwise_8x8" / "depthwise_8x8.asm"
CR_NAMES = {0: "inputs", 1: "kernels", 2: "outputs", 3: "mask"}

SPATIAL = 64  # 8x8

# (num_channels,)
CONFIGS = [8, 16, 32, 64, 128, 160, 256]


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    rng = np.random.RandomState(0)
    rows_data = []

    for ch in CONFIGS:
        label = f"8x8  ch={ch}"
        raw_input = rng.randint(0, 256, ch * SPATIAL, dtype=np.uint8).tobytes()
        raw_kernel = rng.randint(0, 256, ch * 9, dtype=np.uint8).tobytes()
        input_packed = _build_input_data(raw_input, ch)
        kernel_packed = _build_kernel_data(raw_kernel, ch)
        inp = make_tmp_bin(input_packed)
        krn = make_tmp_bin(kernel_packed)
        try:
            app = Depthwise8x8App(
                inst_path=bin_path,
                input_path=inp,
                kernel_path=krn,
                output_path=None,
                channels=ch,
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
