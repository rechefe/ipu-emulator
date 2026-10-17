"""A family's own tests (ipu_kernel_family()'s :tests): they see the family's
registry and every template's files, and with tests_tree the kernel tree, as
a conformance or routing test does."""

from __future__ import annotations

from pathlib import Path

from ipu_harness.family import load_entry, read_registry
from ipu_harness.registry import Registry, kernel_tree

FAMILY = Path(__file__).parent


def test_every_entry_loads_from_the_family_files():
    registry = FAMILY / "registry.toml"
    entries = read_registry(registry)
    assert [e.name for e in entries] == ["scale_r2", "scale_r8_fp32", "scale_r3_half"]
    for entry in entries:
        assert load_entry(registry, entry).asm.is_file()  # the template's program


def test_the_kernel_tree_routes_and_this_fixture_stays_out_of_it():
    registry = Registry.load(kernel_tree())
    assert not registry.invalid
    assert "scale" in registry.manifests and "scale_r2" not in registry.manifests
