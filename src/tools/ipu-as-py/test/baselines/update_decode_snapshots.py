"""Rewrite the decode snapshots from the current code.

    bazel run //src/tools/ipu-as-py:update_decode_snapshots

Run it only in a PR that changes field names, emulator field maps or enum
descriptors on purpose, and say why in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import decode_snapshots


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {decode_snapshots.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    decode_snapshots.write_all(Path(workspace))
    print(f"Wrote {', '.join(decode_snapshots.SNAPSHOTS)} in {decode_snapshots.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
