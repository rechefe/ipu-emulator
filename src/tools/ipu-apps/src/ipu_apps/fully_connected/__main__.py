"""Debug runner for the fully-connected app.

Usage::

    bazel run //src/tools/ipu-apps:fully_connected -- --dtype INT8
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.fully_connected import FullyConnectedApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fully-connected with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "FP8_E4M3", "FP8_E5M2"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    # Both are relative to runfiles root (bazel run sets cwd to runfiles)
    inst_path = Path(os.environ["FC_INST_BIN"])
    data_dir = Path(os.environ["FC_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = FullyConnectedApp(
        inst_path=inst_path,
        inputs_path=data_dir / dtype_dir / f"inputs_{dtype_dir}.bin",
        weights_path=data_dir / dtype_dir / f"weights_{dtype_dir}.bin",
        output_path=args.output,
        dtype=args.dtype,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(f"Finished in {cycles} cycles")


if __name__ == "__main__":
    main()
