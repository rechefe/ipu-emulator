"""Debug runner for the concat-y-and-z app.

Usage::

    bazel run //src/tools/ipu-apps:concat_yz -- --stage stage3
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.concat_yz import ConcatYzApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run concat-y-and-z with debug CLI")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage4"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    # Both are relative to runfiles root (bazel run sets cwd to runfiles)
    inst_path = Path(os.environ["CONCAT_YZ_INST_BIN"])
    data_dir = Path(os.environ["CONCAT_YZ_DATA_DIR"]) / args.stage

    app = ConcatYzApp(
        inst_path=inst_path,
        inputs_path=data_dir / "y_in_int8.bin",
        z_path=data_dir / "z_in_int8.bin",
        output_path=args.output,
        stage=args.stage,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
