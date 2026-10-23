"""Rewrite the generated-artifact snapshots from the current code.

    bazel run //src/ipu-gen:update_generated_artifacts

Run it only in a PR that changes a generated artifact on purpose, and say why
in the PR (Code conventions, "Baselines").
"""

from __future__ import annotations

import sys

import baseline_support
import generated_artifacts


def main() -> int:
    root = baseline_support.workspace(generated_artifacts.UPDATE_COMMAND)
    if root is None:
        return 2
    count = generated_artifacts.write_all(root)
    print(f"Wrote {count} artifacts to {generated_artifacts.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
