"""Rewrite the SV package golden from the current code.

    bazel run //src/tools/ipu-as-py:update_sv_golden

It does not touch the localparam table in test_sv_package_golden.py: a width
change must be an explicit edit there, so review sees it (decision D6).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ipu_gen import sv_package

GOLDEN_PATH = "src/tools/ipu-as-py/test/golden/ipu_instr_pkg.sv"
UPDATE_COMMAND = "bazel run //src/tools/ipu-as-py:update_sv_golden"


def render() -> str:
    """The package exactly as the downstream repo gets it."""
    return sv_package.render_sv_package()


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {UPDATE_COMMAND}", file=sys.stderr)
        return 2
    (Path(workspace) / GOLDEN_PATH).write_text(render(), encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
