"""Rewrite the generated-artifact snapshots from the current code.

    bazel run //src/tools/ipu-as-py:update_generated_artifacts

Run it only in a PR that changes a generated artifact on purpose, and say why
in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import generated_artifacts


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {generated_artifacts.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    count = generated_artifacts.write_all(Path(workspace))
    print(f"Wrote {count} artifacts to {generated_artifacts.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
