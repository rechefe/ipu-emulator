"""Debug runner for the standard 16->16 channel convolution app (8x8)."""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.conv_8x8_16to16 import Conv8x8_16to16App


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8x8 16->16ch std conv with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=10_000_000)
    args = parser.parse_args()

    inst_path = Path(os.environ["CONV8X8_16TO16_INST_BIN"])
    data_dir = Path(os.environ["CONV8X8_16TO16_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = Conv8x8_16to16App(
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
