"""Rewrite the ISA package's derived files from the current ``ipu_isa``.

    bazel run //src/tools/ipu-isa:update_isa_baselines

Writes the view snapshot (``spec_snapshot.SNAPSHOT_PATH``) and the F-1 table in
``faults_table.FAULTS_PAGE``. ``tools/update_baselines.sh`` runs it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import faults_table
import spec_snapshot

from ipu_isa.view import isa


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print("Run this through Bazel: bazel run //src/tools/ipu-isa:update_isa_baselines")
        return 2
    root = Path(workspace)
    view = isa()
    (root / spec_snapshot.SNAPSHOT_PATH).write_text(spec_snapshot.render(view), encoding="utf-8")
    page = root / faults_table.FAULTS_PAGE
    page.write_text(
        faults_table.with_table(page.read_text(encoding="utf-8"), view), encoding="utf-8"
    )
    print(f"Wrote {spec_snapshot.SNAPSHOT_PATH} and the F-1 table in {faults_table.FAULTS_PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
