"""Debug runner for the split-to-x-and-z app.

Usage::

    bazel run //src/tools/ipu-apps:split_xz -- --stage stage3
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.split_xz import SplitXzApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split-to-x-and-z with debug CLI")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage4"])
    parser.add_argument("--output-z", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    # Both are relative to runfiles root (bazel run sets cwd to runfiles)
    inst_path = Path(os.environ["SPLIT_XZ_INST_BIN"])
    data_dir = Path(os.environ["SPLIT_XZ_DATA_DIR"]) / args.stage

    app = SplitXzApp(
        inst_path=inst_path,
        inputs_path=data_dir / "xz_in_int8.bin",
        output_path=args.output,
        output_z_path=args.output_z,
        stage=args.stage,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
