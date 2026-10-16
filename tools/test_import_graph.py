"""First-party packages import only what ``import_graph.toml`` allows.

Reads every Python source of every package under ``src/tools/*/src/<package>/``
and collects the first-party packages each one imports: ``import x``,
``from x import y`` and ``importlib.import_module("x...")`` with a literal
name, at any depth (a function-level import counts). Each must be the
package itself or listed in the package's row of the table.

The check is static, so it sees an import whatever the Bazel deps of the test
happen to be. ``//src/tools/ipu-isa:test_purity`` keeps the complementary
run-time check for ``ipu_isa``: its modules import with only ``ipu_isa_lib`` on
the path.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

TABLE = Path("tools/import_graph.toml")
SOURCE_ROOT = Path("src/tools")


def load_table(path: Path = TABLE) -> dict[str, frozenset[str]]:
    with path.open("rb") as f:
        allowed = tomllib.load(f)["allowed"]
    return {package: frozenset(deps) for package, deps in allowed.items()}


def find_packages(root: Path = SOURCE_ROOT) -> dict[str, Path]:
    """``{package: directory}`` for every ``<root>/*/src/<package>/``."""
    return {path.name: path for path in sorted(root.glob("*/src/*")) if path.is_dir()}


def imported_modules(source: str, filename: str = "<string>") -> Iterator[tuple[int, str]]:
    """``(line, dotted name)`` of each absolute import in ``source``."""
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if (
                name == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and not node.args[0].value.startswith(".")
            ):
                yield node.lineno, node.args[0].value


def violations(
    packages: Mapping[str, Path], table: Mapping[str, frozenset[str]]
) -> list[str]:
    """One line per import of a first-party package that ``table`` forbids."""
    found = []
    for package, directory in sorted(packages.items()):
        allowed = table[package] | {package}
        for path in sorted(directory.rglob("*.py")):
            for line, module in imported_modules(path.read_text(), str(path)):
                top = module.split(".")[0]
                if top in packages and top not in allowed:
                    found.append(f"{path}:{line}: {package} imports {module}")
    return found


TABLE_ROWS = load_table()
PACKAGES = find_packages()


def test_every_package_has_a_row():
    """A package with no row, or a row with no package sources, is a mistake:
    add the row, or the package's ``py_sources`` to this test's data."""
    assert sorted(PACKAGES) == sorted(TABLE_ROWS)


def test_rows_name_known_packages():
    unknown = {
        package: sorted(deps - TABLE_ROWS.keys())
        for package, deps in TABLE_ROWS.items()
        if deps - TABLE_ROWS.keys()
    }
    assert unknown == {}


def test_every_package_has_sources():
    for package, directory in PACKAGES.items():
        assert any(directory.rglob("*.py")), package


def test_only_allowed_imports():
    assert violations(PACKAGES, TABLE_ROWS) == []


# The checker itself, on planted packages.


def _plant(root: Path, files: Mapping[str, str]) -> dict[str, Path]:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return find_packages(root)


@pytest.mark.parametrize(
    "text",
    [
        "import pkg_b\n",
        "import pkg_b.sub as s\n",
        "from pkg_b import x\n",
        "def f():\n    from pkg_b.sub import x\n",
        "import importlib\nimportlib.import_module('pkg_b.sub')\n",
        "from importlib import import_module\nimport_module('pkg_b')\n",
    ],
)
def test_a_forbidden_import_is_reported(tmp_path, text):
    packages = _plant(
        tmp_path,
        {"a/src/pkg_a/mod.py": text, "b/src/pkg_b/__init__.py": ""},
    )
    table = {"pkg_a": frozenset(), "pkg_b": frozenset()}
    [found] = violations(packages, table)
    assert found.startswith(str(tmp_path / "a/src/pkg_a/mod.py"))
    assert "pkg_a imports pkg_b" in found


def test_allowed_self_relative_and_third_party_imports_pass(tmp_path):
    packages = _plant(
        tmp_path,
        {
            "a/src/pkg_a/mod.py": (
                "import os\nimport numpy\nimport pkg_a.other\nfrom . import other\n"
                "from pkg_b import x\nimport importlib\nimportlib.import_module('.other', 'pkg_a')\n"
            ),
            "b/src/pkg_b/__init__.py": "",
        },
    )
    table = {"pkg_a": frozenset({"pkg_b"}), "pkg_b": frozenset()}
    assert violations(packages, table) == []
