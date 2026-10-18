"""No module imports a ``_``-prefixed name from another ``ipu_debug`` module (Q66).

The debugger frontends share code through public modules (``ipu_debug.session``,
``ipu_debug.format``); reaching into ``ipu_debug.cli`` or ``ipu_debug.tui`` for a
private helper is what this test stops.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Sources scanned: this package (library and tests) and the harness that launches it.
_PACKAGE = Path(__file__).parent.parent
_SCANNED = (
    _PACKAGE / "src",
    _PACKAGE / "test",
    _PACKAGE.parent / "ipu-harness" / "src",
)


def _module_name(path: Path) -> str | None:
    """``ipu_debug.<name>`` for a library file, else None."""
    if path.parent.name == "ipu_debug" and path.parent.parent.name == "src":
        return "ipu_debug" if path.stem == "__init__" else f"ipu_debug.{path.stem}"
    return None


def private_imports(source: str, module: str | None = None) -> list[str]:
    """Each ``from ipu_debug[.x] import _name`` in *source*, as ``line: text``.

    An import from *module* itself (the file's own module) is allowed.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        target = node.module or ""
        if not (target == "ipu_debug" or target.startswith("ipu_debug.")) or target == module:
            continue
        for alias in node.names:
            if alias.name.startswith("_") and not alias.name.startswith("__"):
                found.append(f"{node.lineno}: from {target} import {alias.name}")
    return found


def _python_files() -> list[Path]:
    files = sorted(path for root in _SCANNED if root.is_dir() for path in root.rglob("*.py"))
    assert files, f"no sources found under {_SCANNED}"
    return files


def test_no_private_cross_module_imports():
    offenders = [
        f"{path}:{hit}"
        for path in _python_files()
        for hit in private_imports(path.read_text(encoding="utf-8"), _module_name(path))
    ]
    assert not offenders, "private cross-module imports:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "planted",
    [
        "from ipu_debug.cli import _parse_int\n",
        "from ipu_debug.tui import _resolved_key\n",
        "def f():\n    from ipu_debug.cli import (\n        VISIBLE_ALPHABET,\n        _tokenize,\n    )\n",
    ],
)
def test_a_planted_private_import_is_reported(planted):
    assert private_imports(planted, "ipu_debug.session")


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("from ipu_debug.cli import DebugCLI\n", None),
        ("from ipu_debug.cli import _parse_int\n", "ipu_debug.cli"),
        ("from ipu_debug.session import __all__\n", None),
        ("from ipu_emu.ipu import _private\n", None),
    ],
)
def test_public_own_module_and_other_package_imports_pass(source, module):
    assert private_imports(source, module) == []
