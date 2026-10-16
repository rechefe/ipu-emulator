"""Rewrite the SV package golden from the current code.

    bazel run //src/tools/ipu-as-py:update_sv_golden

It does not touch the localparam table in test_sv_package_golden.py: a width
change must be an explicit edit there, so review sees it (decision D6).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ipu_as import gen_codegen

GOLDEN_PATH = "src/tools/ipu-as-py/test/golden/ipu_instr_pkg.sv"


def render() -> str:
    """The package as the downstream repo gets it, with the commit unstamped."""
    context = gen_codegen.build_codegen_context(gen_codegen.UNKNOWN_SOURCE_COMMIT)
    return gen_codegen.render_template("ipu_instr_pkg.sv.j2", context)


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print("Run this through Bazel: bazel run //src/tools/ipu-as-py:update_sv_golden",
              file=sys.stderr)
        return 2
    (Path(workspace) / GOLDEN_PATH).write_text(render(), encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
