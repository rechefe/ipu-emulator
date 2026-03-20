"""Debug runner for the standard 8->16 channel convolution app (64x64)."""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.conv_64x64_8to16 import Conv64x64_8to16App


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 64x64 8->16ch std conv with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=10_000_000)
    args = parser.parse_args()

    inst_path = Path(os.environ["CONV64_8TO16_INST_BIN"])
    data_dir = Path(os.environ["CONV64_8TO16_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = Conv64x64_8to16App(
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
