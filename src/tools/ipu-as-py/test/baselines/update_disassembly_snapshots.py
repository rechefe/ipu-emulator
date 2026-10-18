"""Rewrite the disassembly snapshots from the current tree.

    bazel run //src/tools/ipu-as-py:update_disassembly_snapshots

Run it only in a PR that changes disassembly text on purpose (for example a
mnemonic rename), and say why in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import sys

import baseline_support
import disassembly


def main() -> int:
    root = baseline_support.workspace(disassembly.UPDATE_COMMAND)
    if root is None:
        return 2
    count = disassembly.write_all(root)
    print(f"Wrote {count} snapshots to {disassembly.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
