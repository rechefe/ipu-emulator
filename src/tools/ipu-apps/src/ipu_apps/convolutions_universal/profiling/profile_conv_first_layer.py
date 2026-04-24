"""Profile memory access patterns for conv_first_layer (256x256x3 -> 128x128x16, stride 2).

This app has a fixed configuration so only one run is needed.
CR0-CR2 are the three input channels (CHW layout), CR3=kernels, CR4=mask,
CR5=outputs, CR6=temp.

Run from the repo root:
    PYTHONPATH=src/tools/ipu-emu-py/src:src/tools/ipu-common/src:src/tools/ipu-apps/src:src/tools/ipu-as-py/src \
        python -m ipu_apps.convolutions_universal.profiling.profile_conv_first_layer
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ipu_apps.convolutions_universal.conv_first_layer import (
    ConvFirstLayerApp,
    IN_ROWS, IN_COLS, IN_CHANNELS, OUT_CHANNELS,
)
from ipu_apps.convolutions_universal.profiling._utils import (
    assemble_if_needed, cleanup, make_tmp_bin, print_profile_table, run_profile_safe,
)

ASM = Path(__file__).resolve().parents[1] / "conv_first_layer" / "conv_first_layer.asm"

# CR0-CR2: the three input channels; CR3: kernels; CR4: mask; CR5: outputs; CR6: temp
CR_NAMES = {
    0: "input_ch0",
    1: "input_ch1",
    2: "input_ch2",
    3: "kernels",
    4: "mask",
    5: "outputs",
    6: "temp",
}


def main() -> None:
    bin_path = assemble_if_needed(ASM)
    rng = np.random.RandomState(0)

    input_bytes = rng.randint(0, 256, IN_ROWS * IN_COLS * IN_CHANNELS, dtype=np.uint8).tobytes()
    kernel_bytes = rng.randint(0, 256, OUT_CHANNELS * IN_CHANNELS * 9, dtype=np.uint8).tobytes()
    inp = make_tmp_bin(input_bytes)
    krn = make_tmp_bin(kernel_bytes)

    try:
        app = ConvFirstLayerApp(
            inst_path=bin_path,
            input_path=inp,
            kernel_path=krn,
            output_path=None,
        )
        label = f"{IN_ROWS}x{IN_COLS}x{IN_CHANNELS} -> {OUT_CHANNELS} (stride 2)"
        result = run_profile_safe(app, CR_NAMES)
    except Exception as exc:
        result = str(exc)
        label = "conv_first_layer"
    finally:
        cleanup(inp, krn)

    print_profile_table(
        ["input_ch0", "input_ch1", "input_ch2", "kernels", "mask", "outputs", "temp"],
        [(label, result)],
    )


if __name__ == "__main__":
    main()
