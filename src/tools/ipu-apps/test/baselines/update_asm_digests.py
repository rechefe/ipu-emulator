"""Rewrite asm_digests.json from the current tree.

    bazel run //src/tools/ipu-apps:update_asm_digests

Run it only in a PR that changes assembled output on purpose, and say why in
the PR; a refactor must leave the manifest untouched (decision D3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import asm_corpus


def main() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        print(f"Run this through Bazel: {asm_corpus.UPDATE_COMMAND}", file=sys.stderr)
        return 2
    root = Path(workspace)
    manifest = asm_corpus.compute_manifest(root)
    asm_corpus.write_manifest(root, manifest)
    print(f"Wrote {len(manifest)} entries to {asm_corpus.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
