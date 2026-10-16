"""``ipu_isa`` imports nothing from the other IPU packages (Q31, Q62).

The ISA package is the one every other package depends on, so it must not
depend on any of them, not even through a function-level import that only works
because a caller already has the assembler on the path. The Bazel target of
this test depends on ``ipu_isa_lib`` alone, so Bazel enforces the same rule:
an import of another package fails here with ``ModuleNotFoundError``.
"""

from __future__ import annotations

import json
import os
import pkgutil
import subprocess
import sys

import pytest

import ipu_isa

FORBIDDEN = ("ipu_as", "ipu_emu", "ipu_gen", "ipu_apps")

MODULES = sorted(
    f"ipu_isa.{module.name}" for module in pkgutil.iter_modules(ipu_isa.__path__)
)


def _forbidden_packages_loaded(modules: list[str], extra_path: str | None = None) -> list[str]:
    """Import ``modules`` in a fresh interpreter (this one's ``sys.path``, plus
    ``extra_path`` first); return the forbidden packages it ended up loading."""
    path = ([extra_path] if extra_path else []) + sys.path
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"print(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}} & {set(FORBIDDEN)!r})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(path)},
    )
    return json.loads(result.stdout)


def test_every_module_is_found():
    assert {"ipu_isa.instruction_spec", "ipu_isa.registers"} <= set(MODULES)


@pytest.mark.parametrize("module", MODULES)
def test_no_module_imports_another_ipu_package(module):
    assert _forbidden_packages_loaded([module]) == []


def test_a_planted_import_is_caught(tmp_path):
    """The check sees a module that imports another IPU package."""
    (tmp_path / "ipu_as").mkdir()
    (tmp_path / "ipu_as" / "__init__.py").write_text("")
    (tmp_path / "planted_isa_module.py").write_text("import ipu_as\n")
    assert _forbidden_packages_loaded(["planted_isa_module"], str(tmp_path)) == ["ipu_as"]
