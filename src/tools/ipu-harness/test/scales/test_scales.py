"""A family's own tests (ipu_kernel_family()'s :tests): they see the family's
registry and every template's files, as a routing or conformance test does."""

from __future__ import annotations

from pathlib import Path

from ipu_harness.family import load_entry, read_registry

FAMILY = Path(__file__).parent


def test_every_entry_loads_from_the_familys_files():
    registry = FAMILY / "registry.toml"
    entries = read_registry(registry)
    assert [e.name for e in entries][:2] == ["scale_r2", "scale_r8_fp32"]
    for entry in entries:
        manifest = load_entry(registry, entry)
        assert manifest.asm.is_file()  # the template's program is in the runfiles
