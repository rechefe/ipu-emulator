"""Profile memory access patterns for pointwise_conv_universal.

Sweeps spatial sizes and channel counts, printing peak lookahead rows
per memory region (inputs, kernels, outputs, mask).

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_pointwise_universal
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ipu_apps.convolutions_universal.pointwise_conv_universal import PointwiseConvUniversalApp
from ipu_apps.convolutions_universal.profiling._utils import (
    cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

BIN = Path(__file__).resolve().parents[1] / "pointwise_conv_universal" / "pointwise_conv_universal.bin"
CR_NAMES = {0: "inputs", 1: "kernels", 2: "mask", 3: "outputs"}

# (rows, cols, in_channels, out_channels)
CONFIGS = [
    (64, 64,  16,  32),
    (64, 64,  32,  64),
    (64, 64,  64, 128),
    (64, 64, 128,  64),
    (64, 64,   8,  32),
    (16, 16,  16,  32),
    (32, 32,  16,  32),
    (128, 128, 16, 32),
]


def main() -> None:
    rng = np.random.RandomState(0)
    rows_data = []

    for rows, cols, ic, oc in CONFIGS:
        label = f"{rows}x{cols}  ic={ic}  oc={oc}"
        input_bytes = rng.randint(0, 256, rows * cols * ic, dtype=np.uint8).tobytes()
        kernel_bytes = rng.randint(0, 256, oc * ic, dtype=np.uint8).tobytes()
        inp = make_tmp_bin(input_bytes)
        krn = make_tmp_bin(kernel_bytes)
        try:
            app = PointwiseConvUniversalApp(
                inst_path=BIN,
                input_path=inp,
                kernel_path=krn,
                output_path=None,
                dtype="INT8",
                rows=rows, cols=cols,
                in_channels=ic, out_channels=oc,
            )
            result = run_profile_safe(app, CR_NAMES)
        except Exception as exc:
            result = str(exc)
        finally:
            cleanup(inp, krn)
        rows_data.append((label, result))

    print_profile_table(["inputs", "kernels", "mask", "outputs"], rows_data)


if __name__ == "__main__":
    main()
