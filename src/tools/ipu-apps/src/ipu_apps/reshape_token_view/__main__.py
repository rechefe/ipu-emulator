"""Debug runner for the reshape-to-token-view app.

Usage::

    bazel run //src/tools/ipu-apps:reshape_token_view -- --stage stage3
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.reshape_token_view import ReshapeTokenViewApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reshape-to-token-view with debug CLI")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage4"])
    parser.add_argument("--direction", default="t2c", choices=["t2c", "c2t"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=20_000_000)
    args = parser.parse_args()

    # Both are relative to runfiles root (bazel run sets cwd to runfiles)
    inst_path = Path(os.environ["RESHAPE_INST_BIN"])
    data_dir = Path(os.environ["RESHAPE_DATA_DIR"]) / args.stage

    app = ReshapeTokenViewApp(
        inst_path=inst_path,
        inputs_path=data_dir / f"{args.direction}_in_int8.bin",
        output_path=args.output,
        stage=args.stage,
        direction=args.direction,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
