"""Rewrite asm_digests.json from the current tree.

    bazel run //src/tools/ipu-apps:update_asm_digests

Run it only in a PR that changes assembled output on purpose, and say why in
the PR; a refactor must leave the manifest untouched (Code conventions, "Baselines").
"""

from __future__ import annotations

import sys

import asm_corpus
import baseline_support


def main() -> int:
    root = baseline_support.workspace(asm_corpus.UPDATE_COMMAND)
    if root is None:
        return 2
    manifest = asm_corpus.compute_manifest(root)
    asm_corpus.write_manifest(root, manifest)
    print(f"Wrote {len(manifest)} entries to {asm_corpus.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
