"""Debug runner for the 16x16 256->128 channel convolution app.

Usage::

    bazel run //src/tools/ipu-apps:conv_16x16_256to128 -- --dtype INT8
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.conv_16x16_256to128 import Conv16x16x256to128App


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 16x16 256->128 conv with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=5_000_000)
    args = parser.parse_args()

    inst_path = Path(os.environ["CONV16X16_256TO128_INST_BIN"])
    data_dir = Path(os.environ["CONV16X16_256TO128_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = Conv16x16x256to128App(
        inst_path=inst_path,
        input_path=data_dir / dtype_dir / f"input_{dtype_dir}.bin",
        kernel_path=data_dir / dtype_dir / f"kernel_{dtype_dir}.bin",
        output_path=args.output,
        dtype=args.dtype,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(f"Finished in {cycles} cycles")


if __name__ == "__main__":
    main()
