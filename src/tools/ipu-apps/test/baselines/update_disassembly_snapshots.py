"""Rewrite the disassembly snapshots from the current tree.

    bazel run //src/tools/ipu-apps:update_disassembly_snapshots

Run it only in a PR that changes disassembly text on purpose (for example a
mnemonic rename), and say why in the PR (decision D3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import disassembly


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {disassembly.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    count = disassembly.write_all(Path(workspace))
    print(f"Wrote {count} snapshots to {disassembly.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
