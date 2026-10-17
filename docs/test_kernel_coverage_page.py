"""The Kernel coverage page's tables come from the registries (E09-10).

Every routed kernel and every ``ipu_apps`` kernel is on the generated page, a
kernel file that does not load is listed rather than dropped, and no other
page keeps a coverage list of its own.
"""

from __future__ import annotations

import re
from pathlib import Path

from docs import kernel_coverage_page as page
from ipu_emu.errors import ConfigurationError
from ipu_harness.registry import Registry, default_registry

PAGE = Path("docs/content/explanation/kernel-coverage.md")
KERNELS_INDEX = Path("docs/content/kernels/index.md")


def test_the_snippets_hold_every_kernel_of_both_registries(tmp_path: Path) -> None:
    from ipu_apps.kernel_registry.registry import load

    assert page.main([str(tmp_path)]) == 0
    kernels = (tmp_path / page.KERNELS_FILE).read_text(encoding="utf-8")
    registry = default_registry()
    assert registry.manifests, "the kernel tree is missing from the test data"
    for m in registry.manifests.values():
        assert f"| `{m.op}` | `{m.name}` | `{m.label}` |" in kernels
    apps = (tmp_path / page.IPU_APPS_FILE).read_text(encoding="utf-8")
    specs = load().specs
    assert specs
    for spec in specs:
        assert f"`{spec.name}`" in apps


def test_a_kernel_file_that_did_not_load_is_listed() -> None:
    registry = default_registry()
    broken = Registry(
        manifests=registry.manifests,
        invalid={Path("kernels/x/y/kernel.toml"): ConfigurationError("bad | key")},
    )
    table = page.kernels_table(broken)
    assert "- `kernels/x/y/kernel.toml`: bad \\| key" in table
    assert "did not load" not in page.kernels_table(Registry(manifests=registry.manifests))


def test_the_page_includes_both_snippets() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for name in (page.KERNELS_FILE, page.IPU_APPS_FILE):
        assert f'--8<-- "generated/kernel-coverage/{name}"' in text


def test_no_page_keeps_a_coverage_table_by_hand() -> None:
    # A hand-kept list of operations and their kernels is a second copy of
    # the registry, and goes stale (the old app-coverage page's did).
    for path in (PAGE, KERNELS_INDEX):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\|\s*Operations?\s*\|\s*Kernels?\s*\|", text, re.M | re.I), path
