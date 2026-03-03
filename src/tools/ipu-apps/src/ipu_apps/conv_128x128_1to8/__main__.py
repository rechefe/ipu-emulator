"""Debug runner for the standard 1->8 channel convolution app.

Usage::

    bazel run //src/tools/ipu-apps:conv_128x128_1to8 -- --dtype INT8
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.conv_128x128_1to8 import Conv128x128_1to8App


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1->8ch std conv with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    inst_path = Path(os.environ["CONV1TO8_INST_BIN"])
    data_dir = Path(os.environ["CONV1TO8_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = Conv128x128_1to8App(
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
