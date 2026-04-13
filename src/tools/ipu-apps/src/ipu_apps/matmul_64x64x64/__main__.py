"""Debug runner for the matmul_64x64x64 app.

Usage::

    bazel run //src/tools/ipu-apps:matmul_64x64x64 -- --dtype INT8
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.matmul_64x64x64 import MatMul64x64x64App


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matmul_64x64x64 with debug CLI")
    parser.add_argument("--dtype", default="INT8", choices=["INT8", "E4", "E5"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    inst_path = Path(os.environ["MATMUL_64X64X64_INST_BIN"])
    data_dir = Path(os.environ["MATMUL_64X64X64_DATA_DIR"])
    dtype_dir = args.dtype.lower()

    app = MatMul64x64x64App(
        inst_path=inst_path,
        input_path=data_dir / dtype_dir / f"input_{dtype_dir}.bin",
        weights_path=data_dir / dtype_dir / f"weights_{dtype_dir}.bin",
        output_path=args.output,
        dtype=args.dtype,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(f"Finished in {cycles} cycles")


if __name__ == "__main__":
    main()
