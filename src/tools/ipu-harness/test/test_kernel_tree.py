"""kernels/BUILD.bazel's kernel list and registries, //kernels:tree and //kernels:manifest
(E06-13, E06-17)."""

from __future__ import annotations

import json
from pathlib import Path

from ipu_harness import registry_bzl
from ipu_harness.family import BZL_NAME, read_registry
from ipu_harness.manifest_json import build
from ipu_harness.registry import Registry, kernel_tree

# Bazel runs the test from the runfiles root of the main repository.
TARGETS = Path.cwd() / "kernels" / "kernel_targets.json"
#: The checkout's kernels/ (the BUILD file is a symlink into it).
SOURCE = (Path.cwd() / "kernels" / "BUILD.bazel").resolve().parent

#: //src/tools/ipu-apps:kernel_manifest's keys, which tools read.
LEGACY_KEYS = {"kernels", "families", "operations", "query", "skipped"}
LEGACY_KERNEL_KEYS = {"op", "variant", "tags", "family", "asm", "targets", "cases"}


def _targets() -> dict:
    return json.loads(TARGETS.read_text(encoding="utf-8"))


def test_the_registries_name_every_registry_toml_and_its_entries():
    for path in sorted(SOURCE.glob("*/registry.toml")):
        assert registry_bzl.check(path, path.with_name(BZL_NAME)) is None
        listed = {
            name
            for name, kernel in _targets()["kernels"].items()
            if kernel.get("registry") == str(path.relative_to(SOURCE.parent))
        }
        assert listed == {e.name for e in read_registry(path)}, (
            f"add {path.parent.name} to REGISTRIES in kernels/BUILD.bazel, or run update_registry_bzl"
        )
    with_registry = {k["family"] for k in _targets()["kernels"].values() if "registry" in k}
    assert with_registry <= {p.parent.name for p in SOURCE.glob("*/registry.toml")}


def test_the_tree_holds_every_kernels_files():
    registry = Registry.load(kernel_tree())
    assert not registry.invalid
    assert {*registry.manifests, *registry.templates} == set(_targets()["kernels"])


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
    assert manifest["families"]["elementwise"] == {"suite": "//kernels/elementwise:all"}
    # A registry entry (E07-02's projections) is a kernel of its family's package.
    entry = manifest["kernels"]["proj_qkv_144_p4"]
    assert entry["targets"]["run"] == "//kernels/projections:proj_qkv_144_p4"
    assert manifest["families"]["projections"] == {"suite": "//kernels/projections:all"}


def test_every_kernel_has_design_notes_and_an_abi_table():
    """The kernel page's data (E06-21): the .asm header (Q79) and [abi]."""
    manifest = build(_targets(), Registry.load(kernel_tree()))
    for name, kernel in manifest["kernels"].items():
        assert kernel["design_notes"], f"{kernel['asm']}: start it with a {{#- ... -#}} header"
        assert kernel["abi"], name
    scale = manifest["kernels"]["scale"]
    assert scale["design_notes"].startswith("scale.asm -- y = x * s")
    assert [(r["name"], r["register"], r["kind"]) for r in scale["abi"]] == [
        ("x", "CR2", "region"),
        ("y", "CR3", "region"),
        ("rows", "CR4", "int"),
        ("s", "CR5", "element scalar"),
    ]
    assert scale["params"]["rows"] == {"type": "int", "unit": "count", "min": 1, "max": 256}
    assert scale["tolerance"]["e4"] == {"y": "EXACT"}  # an accumulator output: compared as FP32
