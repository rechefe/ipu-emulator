"""Rewrite the decode snapshots from the current code.

    bazel run //src/ipu-as:update_decode_snapshots

Run it only in a PR that changes field names, emulator field maps or enum
descriptors on purpose, and say why in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import sys

import baseline_support
import decode_snapshots


def main() -> int:
    root = baseline_support.workspace(decode_snapshots.UPDATE_COMMAND)
    if root is None:
        return 2
    decode_snapshots.write_all(root)
    print(f"Wrote {', '.join(decode_snapshots.SNAPSHOTS)} in {decode_snapshots.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
