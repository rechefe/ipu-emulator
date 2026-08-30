"""Debug runner for the mixer-residual app.

Usage::

    bazel run //src/tools/ipu-apps:residual_mixer -- --stage stage3
"""

import argparse
import os
from pathlib import Path

from ipu_emu.debug_cli import debug_prompt

from ipu_apps.residual_mixer import ResidualMixerApp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixer-residual with debug CLI")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage4"])
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=2_000_000)
    args = parser.parse_args()

    # Both are relative to runfiles root (bazel run sets cwd to runfiles)
    inst_path = Path(os.environ["RESIDUAL_MIXER_INST_BIN"])
    data_dir = Path(os.environ["RESIDUAL_MIXER_DATA_DIR"]) / args.stage

    app = ResidualMixerApp(
        inst_path=inst_path,
        inputs_path=data_dir / "skip_in_int8.bin",
        branch_path=data_dir / "branch_in_int8.bin",
        output_path=args.output,
        stage=args.stage,
        gamma=args.gamma,
    )
    state, cycles = app.run(max_cycles=args.max_cycles, debug_callback=debug_prompt)
    print(state.stats.format_summary())


if __name__ == "__main__":
    main()
