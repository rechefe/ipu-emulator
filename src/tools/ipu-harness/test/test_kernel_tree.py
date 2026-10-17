"""kernels/BUILD.bazel's kernel list, //kernels:tree and //kernels:manifest (E06-13)."""

from __future__ import annotations

import json
from pathlib import Path

from ipu_harness.manifest_json import build
from ipu_harness.registry import Registry, kernel_tree

# Bazel runs the test from the runfiles root of the main repository.
TARGETS = Path.cwd() / "kernels" / "kernel_targets.json"

#: //src/tools/ipu-apps:kernel_manifest's keys, which tools read.
LEGACY_KEYS = {"kernels", "families", "operations", "query", "skipped"}
LEGACY_KERNEL_KEYS = {"op", "variant", "tags", "family", "asm", "targets", "cases"}


def _targets() -> dict:
    return json.loads(TARGETS.read_text(encoding="utf-8"))


def test_the_tree_holds_every_kernels_files():
    registry = Registry.load(kernel_tree())
    assert not registry.invalid
    assert set(registry.manifests) == set(_targets()["kernels"])


def test_the_manifest_has_the_legacy_keys():
    manifest = build(_targets(), Registry.load(kernel_tree()))
    assert set(manifest) == LEGACY_KEYS
    scale = manifest["kernels"]["scale"]
    assert LEGACY_KERNEL_KEYS <= set(scale)
    assert scale["targets"] == {
        "run": "//kernels/elementwise/scale:scale",
        "test": "//kernels/elementwise/scale:scale",
        "nightly": "//kernels/elementwise/scale:scale_nightly",
    }
    assert scale["asm"] == "kernels/elementwise/scale/scale.asm"
    assert list(scale["cases"])[:2] == ["default", "one_row"]
    # The legacy shapes tools read: option descriptors (none: a case's values are
    # fixed by its name) and a family's suite label.
    assert scale["cases"]["default"]["options"] == {}
    assert scale["cases"]["default"]["values"] == {"rows": 4, "s": 2.0}
    assert manifest["families"] == {"elementwise": {"suite": "//kernels/elementwise:all"}}
