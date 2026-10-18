"""Rewrite the SV package golden from the current code.

    bazel run //src/ipu-as:update_sv_golden

It does not touch the localparam table in test_sv_package_golden.py: a width
change must be an explicit edit there, so review sees it (CLAUDE.md,
"Generated SystemVerilog Package").
"""

from __future__ import annotations

import sys

import baseline_support

from ipu_gen import sv_package

GOLDEN_PATH = "src/ipu-as/test/baselines/ipu_instr_pkg.sv"
UPDATE_COMMAND = "bazel run //src/ipu-as:update_sv_golden"


def render() -> str:
    """The package exactly as the downstream repo gets it."""
    return sv_package.render_sv_package()


def main() -> int:
    root = baseline_support.workspace(UPDATE_COMMAND)
    if root is None:
        return 2
    (root / GOLDEN_PATH).write_text(render(), encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
