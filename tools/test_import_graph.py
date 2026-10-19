"""First-party packages import only what ``import_graph.toml`` allows.

Reads every Python source of every package under ``src/*/src/<package>/``
and collects the modules each one imports: ``import x``, ``from x import y``
and ``importlib.import_module("x...")`` with a literal name, at any depth (a
function-level import counts). Each first-party one must be the package itself
or listed in the source's ``[allowed]`` row, and a module in the source's
``[deferred]`` row may only be imported inside a function or under
``if TYPE_CHECKING:``. A name in a module's ``[private]`` row may not be
imported, or read as an attribute, outside that module's package.

A row's key is a package or a dotted subpackage (``ipu_kernels.reference``);
a source follows the longest key that is its module or a parent of it, in
each table separately (E06-01).

The check is static, so it sees an import whatever the Bazel deps of the test
happen to be. ``//src/ipu-isa:test_purity`` keeps the complementary
run-time check for ``ipu_isa``: its modules import with only ``ipu_isa_lib`` on
the path.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

TABLE = Path("tools/import_graph.toml")
SOURCE_ROOT = Path("src")

#: Top-level modules with this prefix are first party, whether or not their
#: sources are in this test's data.
FIRST_PARTY_PREFIX = "ipu_"

Table = Mapping[str, frozenset[str]]


SECTIONS = ("allowed", "deferred", "private")


def load_table(path: Path = TABLE, section: str = "allowed") -> dict[str, frozenset[str]]:
    with path.open("rb") as f:
        rows = tomllib.load(f)[section]
    return {key: frozenset(deps) for key, deps in rows.items()}


def find_packages(root: Path = SOURCE_ROOT) -> dict[str, Path]:
    """``{package: directory}`` for every ``<root>/*/src/<package>/``."""
    return {path.name: path for path in sorted(root.glob("*/src/*")) if path.is_dir()}


class Import(NamedTuple):
    line: int
    module: str
    deferred: bool  # in a function body or under `if TYPE_CHECKING:`


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` or ``if <anything>.TYPE_CHECKING:``; a compound
    test such as ``if TYPE_CHECKING and x:`` does not count (fails safe)."""
    name = test.attr if isinstance(test, ast.Attribute) else getattr(test, "id", None)
    return name == "TYPE_CHECKING"


def _literal_import_module(node: ast.Call) -> str | None:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    arg = node.args[0] if node.args else None
    if (
        name == "import_module"
        and isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and not arg.value.startswith(".")
    ):
        return arg.value
    return None


def _imports(node: ast.AST, deferred: bool) -> Iterator[Import]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield Import(node.lineno, alias.name, deferred)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        yield Import(node.lineno, node.module, deferred)
    elif isinstance(node, ast.Call) and (module := _literal_import_module(node)):
        yield Import(node.lineno, module, deferred)
    # Only a body runs later: a function's decorators, defaults and annotations
    # run when the module is imported.
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        later = node.body
    elif isinstance(node, ast.Lambda):
        later = [node.body]
    elif isinstance(node, ast.If) and _is_type_checking(node.test):
        later = node.body
    else:
        later = []
    for child in ast.iter_child_nodes(node):
        yield from _imports(child, deferred or any(child is n for n in later))


def imported_modules(source: str, filename: str = "<string>") -> Iterator[Import]:
    """Each absolute import in ``source``, depth first, roughly in source order."""
    yield from _imports(ast.parse(source, filename), deferred=False)


def module_name(path: Path, directory: Path) -> str:
    """``ipu_kernels.reference.softmax`` for ``.../ipu_kernels/reference/softmax.py``."""
    parts = path.relative_to(directory.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def within(module: str, prefix: str) -> bool:
    """``module`` is ``prefix`` or inside it."""
    return module == prefix or module.startswith(prefix + ".")


def key_of(table: Table, module: str) -> str | None:
    """The longest key of ``table`` that is ``module`` or a parent of it."""
    keys = [k for k in table if within(module, k)]
    return max(keys, key=len) if keys else None


def row(table: Table, module: str) -> frozenset[str]:
    key = key_of(table, module)
    return table[key] if key is not None else frozenset()


def violations(
    packages: Mapping[str, Path], table: Table, deferred: Table | None = None
) -> list[str]:
    """One line per import that ``table`` (``[allowed]``) or ``deferred`` forbids.

    An ``ipu_*`` package with no sources in ``packages`` is reported too: it
    is first party but not wired into this test.
    """
    found = []
    for package, directory in sorted(packages.items()):
        for path in sorted(directory.rglob("*.py")):
            source = module_name(path, directory)
            # A source imports its own key's subtree freely: the whole package
            # under a package row, only the subpackage under a subpackage row.
            allowed = row(table, source) | {key_of(table, source) or package}
            lazy_only = row(deferred or {}, source)
            for imp in imported_modules(path.read_text(), str(path)):
                top = imp.module.split(".")[0]
                where = f"{path}:{imp.line}: {source}"
                if top not in packages and top.startswith(FIRST_PARTY_PREFIX):
                    found.append(
                        f"{where} imports unknown first-party package {top}; "
                        "add its :py_sources to //tools:test_import_graph and its row to "
                        f"{TABLE}"
                    )
                elif top in packages and not any(within(imp.module, a) for a in allowed):
                    found.append(f"{where} imports {imp.module}")
                elif top in lazy_only and not imp.deferred:
                    found.append(f"{where} imports {imp.module} at module level")
    return found


def private_name_uses(packages: Mapping[str, Path], private: Table) -> list[str]:
    """One line per use of a ``[private]`` name outside its module's package:
    ``from module import NAME`` (or ``*``), or any ``<expr>.NAME``. A package's
    sources and its distribution's ``test/`` directory both count as the
    package."""
    found = []
    names = {name: module for module, members in private.items() for name in members}
    for package, directory in sorted(packages.items()):
        tests = directory.parent.parent / "test"
        paths = sorted(directory.rglob("*.py")) + sorted(tests.rglob("*.py"))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(), str(path))):
                if isinstance(node, ast.ImportFrom) and node.module in private:
                    used = [
                        a.name if a.name != "*" else "*"
                        for a in node.names
                        if a.name in private[node.module] or a.name == "*"
                    ]
                elif isinstance(node, ast.Attribute) and node.attr in names:
                    used = [node.attr]
                else:
                    continue
                for name in used:
                    owner = node.module if name == "*" else names[name]
                    if owner.split(".")[0] != package:
                        found.append(f"{path}:{node.lineno}: uses {owner}.{name}")
    return found


TABLE_ROWS = load_table()
DEFERRED_ROWS = load_table(section="deferred")
PRIVATE_ROWS = load_table(section="private")
PACKAGES = find_packages()


def test_every_package_has_a_row():
    """A package with no row, or a row with no package sources, is a mistake:
    add the row, or the package's ``:py_sources`` to this test's data."""
    assert {
        "no row": sorted(PACKAGES.keys() - TABLE_ROWS.keys()),
        "no sources": sorted({k.split(".")[0] for k in TABLE_ROWS} - PACKAGES.keys()),
    } == {"no row": [], "no sources": []}


def test_table_has_only_known_sections():
    with TABLE.open("rb") as f:
        assert sorted(tomllib.load(f)) == sorted(SECTIONS)


def test_rows_name_known_packages():
    unknown = {
        key: sorted(d for d in deps if d.split(".")[0] not in PACKAGES)
        for key, deps in TABLE_ROWS.items()
    }
    assert {k: v for k, v in unknown.items() if v} == {}


def test_deferred_rows_are_valid():
    """A deferred module is a top-level name (the check compares those), and a
    first-party one is also allowed for the key."""
    bad = {}
    for key, modules in DEFERRED_ROWS.items():
        allowed = row(TABLE_ROWS, key)
        for module in modules:
            if "." in module or (module in PACKAGES and module not in allowed):
                bad.setdefault(key, []).append(module)
    assert bad == {}


def test_subpackage_keys_name_directories():
    """A dotted key names a subpackage directory; a typo would match no file."""
    missing = []
    for key in {*TABLE_ROWS, *DEFERRED_ROWS}:
        package, *rest = key.split(".")
        if package not in PACKAGES or not PACKAGES[package].joinpath(*rest).is_dir():
            missing.append(key)
    assert sorted(missing) == []


def test_every_package_has_sources():
    for package, directory in PACKAGES.items():
        assert any(directory.rglob("*.py")), package


def test_only_allowed_imports():
    assert violations(PACKAGES, TABLE_ROWS, DEFERRED_ROWS) == []


def test_private_names_stay_in_their_package():
    assert private_name_uses(PACKAGES, PRIVATE_ROWS) == []


@pytest.mark.parametrize(
    "text",
    [
        "from ipu_isa.instruction_spec import INSTRUCTION_SPEC\n",
        "def f():\n    from ipu_isa.instruction_spec import PSEUDO_INSTRUCTION_SPEC\n",
        "import ipu_isa.instruction_spec as spec\nx = spec.INSTRUCTION_SPEC\n",
        "from ipu_isa.instruction_spec import *\n",
    ],
)
@pytest.mark.parametrize("where", ["as/src/ipu_as/mod.py", "as/test/test_mod.py"])
def test_a_planted_private_name_use_is_reported(tmp_path, text, where):
    packages = _plant(
        tmp_path,
        {
            "as/src/ipu_as/__init__.py": "",
            where: text,
            "isa/src/ipu_isa/view.py": "from ipu_isa.instruction_spec import INSTRUCTION_SPEC\n",
            "isa/test/test_view.py": "from ipu_isa.instruction_spec import INSTRUCTION_SPEC\n",
        },
    )
    [found] = private_name_uses(packages, PRIVATE_ROWS)
    assert found.startswith(str(tmp_path / where))
    assert "ipu_isa.instruction_spec." in found


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
    assert "pkg_a.mod imports pkg_b" in found


def test_an_unwired_first_party_package_is_reported(tmp_path):
    packages = _plant(tmp_path, {"a/src/ipu_a/mod.py": "import ipu_new.sub\n"})
    [found] = violations(packages, {"ipu_a": frozenset()})
    assert "unknown first-party package ipu_new" in found


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


def test_imports_are_marked_deferred_inside_functions_and_type_checking():
    source = (
        "import a\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import b\nelse:\n    import c\n"
        "def f():\n    import d\n"
        "class K:\n    import e\n    def m(self):\n        import f\n"
        "g = lambda: __import__('x') or importlib.import_module('h')\n"
        "@dec(importlib.import_module('i'))\n"
        "def k(x=importlib.import_module('j')):\n    pass\n"
    )
    deferred = {imp.module: imp.deferred for imp in imported_modules(source)}
    assert deferred == {
        "a": False,
        "typing": False,
        "b": True,
        "c": False,
        "d": True,
        "e": False,
        "f": True,
        "h": True,
        "i": False,
        "j": False,
    }


def test_longest_key_wins():
    table = {"p": frozenset({"x"}), "p.sub": frozenset({"y"})}
    assert row(table, "p") == {"x"}
    assert row(table, "p.other.mod") == {"x"}
    assert row(table, "p.sub") == {"y"}
    assert row(table, "p.sub.mod") == {"y"}
    assert row(table, "p.subway") == {"x"}
    assert row(table, "q") == frozenset()


# The E06-01 rules, on planted copies of the new packages with the real table.

_PACKAGES = {
    "harness/src/ipu_harness/__init__.py": "",
    "kernels/src/ipu_kernels/__init__.py": "",
    "kernels/src/ipu_kernels/reference/__init__.py": "",
    "gen/src/ipu_gen/__init__.py": "",
    "emu/src/ipu_emu/__init__.py": "",
    "isa/src/ipu_isa/__init__.py": "",
    "as/src/ipu_as/__init__.py": "",
}


@pytest.mark.parametrize(
    ("file", "text", "message"),
    [
        ("harness/src/ipu_harness/runner.py", "import ipu_gen\n", "imports ipu_gen"),
        ("harness/src/ipu_harness/registry.py", "import ipu_kernels\n", "imports ipu_kernels"),
        # Not even inside a function (Q62): no cycle-dodging lazy imports.
        (
            "harness/src/ipu_harness/registry.py",
            "def load():\n    import ipu_kernels.elementwise.layers\n",
            "imports ipu_kernels.elementwise.layers",
        ),
        ("harness/src/ipu_harness/reference.py", "import torch\n", "torch at module level"),
        (
            "kernels/src/ipu_kernels/reference/softmax.py",
            "from ipu_emu.ipu_state import IpuState\n",
            "imports ipu_emu.ipu_state",
        ),
        (
            "kernels/src/ipu_kernels/reference/softmax.py",
            "import ipu_harness\n",
            "imports ipu_harness",
        ),
        # The rest of ipu_kernels may import ipu_harness, so a reference may not
        # reach it through them.
        (
            "kernels/src/ipu_kernels/reference/softmax.py",
            "from ipu_kernels.pack import rows\n",
            "imports ipu_kernels.pack",
        ),
    ],
)
def test_e06_01_rules_reject(tmp_path, file, text, message):
    packages = _plant(tmp_path, {**_PACKAGES, file: text})
    [found] = violations(packages, TABLE_ROWS, DEFERRED_ROWS)
    assert message in found


def test_e06_01_rules_accept(tmp_path):
    packages = _plant(
        tmp_path,
        {
            **_PACKAGES,
            "harness/src/ipu_harness/reference.py": (
                "import ipu_emu\nimport ipu_isa\nimport importlib\n"
                "def load(entry):\n    import torch\n"
                "    return importlib.import_module(entry['module'])\n"
            ),
            "kernels/src/ipu_kernels/pack/rows.py": "import ipu_harness\n",
            "kernels/src/ipu_kernels/reference/softmax.py": (
                "import ipu_isa\nimport torch\nfrom ipu_kernels.reference import common\n"
            ),
        },
    )
    assert violations(packages, TABLE_ROWS, DEFERRED_ROWS) == []


# Rules of the real table, checked on planted packages.

#: The only packages that may import the debugger: the runner (Q62, E08-03).
DEBUGGER_IMPORTERS = frozenset({"ipu_harness"})


def test_only_the_runner_may_import_the_debugger():
    importers = {package for package, deps in TABLE_ROWS.items() if "ipu_debug" in deps}
    assert importers <= DEBUGGER_IMPORTERS


@pytest.mark.parametrize(
    "text",
    ["import ipu_debug\n", "def f():\n    from ipu_debug.tui import run_debug_tui\n"],
)
def test_the_emulator_importing_the_debugger_is_reported(tmp_path, text):
    packages = _plant(
        tmp_path,
        {
            "ipu-emu/src/ipu_emu/emulator.py": text,
            "ipu-debug/src/ipu_debug/__init__.py": "import ipu_emu\n",
        },
    )
    table = {name: TABLE_ROWS[name] & packages.keys() for name in packages}
    [found] = violations(packages, table)
    assert "ipu_emu.emulator imports ipu_debug" in found
