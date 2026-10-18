"""Nothing outside the engine reaches into ``Ipu`` (E04-11).

Observers, tools and tests see the emulator through public records only:
``IssueFacts``, the ``CycleObserver`` protocol, ranged register and XMEM events
and ``ipu_emu.geometry``. This gate fails on either of:

- ``from ipu_emu.ipu import _name``;
- ``ipu._name`` or ``Ipu._name`` (an attribute starting with one underscore
  read off a name ``ipu`` or ``Ipu``: an engine, the module or the class);

in any Python source or test under ``src/tools/``, except the engine's own
modules: ``ipu.py`` and the modules it is split into (the issuer, the cycle
and the slot handler modules of ``ipu_emu.slots``, E04-14).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

SOURCE_ROOT = Path("src/tools")

#: The engine itself, repo-relative: free to use its own privates.
ENGINE_MODULES = frozenset(
    SOURCE_ROOT / "ipu-emu/src/ipu_emu" / name
    for name in (
        "ipu.py",
        "issue.py",
        "cycle.py",
        *(
            f"slots/{slot}.py"
            for slot in ("aaq", "acc", "brk", "cond", "load", "lr", "mult", "store")
        ),
    )
)

#: Names under which code holds an engine, the engine module or its class.
ENGINE_NAMES = frozenset({"ipu", "Ipu"})


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def engine_private_uses(source: str, filename: str = "<string>") -> Iterator[str]:
    """Each private engine name *source* uses, as ``line: text``."""
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.ImportFrom) and node.module == "ipu_emu.ipu" and not node.level:
            for alias in node.names:
                if _is_private(alias.name):
                    yield f"{node.lineno}: from ipu_emu.ipu import {alias.name}"
        elif (
            isinstance(node, ast.Attribute)
            and _is_private(node.attr)
            and isinstance(node.value, ast.Name)
            and node.value.id in ENGINE_NAMES
        ):
            yield f"{node.lineno}: {node.value.id}.{node.attr}"


def _python_files() -> list[Path]:
    files = sorted(
        path
        for path in SOURCE_ROOT.glob("*/**/*.py")
        if path.parts[3] in ("src", "test") and path not in ENGINE_MODULES
    )
    assert files, f"no sources found under {SOURCE_ROOT}"
    return files


#: Observers: they read no private of any ipu_emu object but their own.
OBSERVER_MODULES = (SOURCE_ROOT / "ipu-emu/src/ipu_emu/alias_observer.py",)


def foreign_private_uses(source: str, filename: str = "<string>") -> Iterator[str]:
    """Each private ``ipu_emu`` import, and each ``x._name`` with ``x`` not ``self``."""
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ipu_emu"):
            for alias in node.names:
                if _is_private(alias.name):
                    yield f"{node.lineno}: from {node.module} import {alias.name}"
        elif (
            isinstance(node, ast.Attribute)
            and _is_private(node.attr)
            and not (isinstance(node.value, ast.Name) and node.value.id == "self")
        ):
            yield f"{node.lineno}: {ast.unparse(node)}"


@pytest.mark.parametrize("path", OBSERVER_MODULES, ids=str)
def test_observers_read_only_public_names(path):
    assert list(foreign_private_uses(path.read_text(encoding="utf-8"), str(path))) == []


@pytest.mark.parametrize(
    "planted",
    [
        "from ipu_emu.ipu import _SLOT_PLANS\n",
        "def f(state):\n    return state.regfile._mutable_exports\n",
        "def f(cycle):\n    return cycle._buffer\n",
    ],
)
def test_a_planted_observer_private_use_is_reported(planted):
    assert list(foreign_private_uses(planted))


def test_engine_modules_exist():
    assert sorted(path for path in ENGINE_MODULES if not path.is_file()) == []


def test_no_engine_privates_outside_the_engine():
    offenders = [
        f"{path}:{hit}"
        for path in _python_files()
        for hit in engine_private_uses(path.read_text(encoding="utf-8"), str(path))
    ]
    assert offenders == []


@pytest.mark.parametrize(
    "planted",
    [
        "from ipu_emu.ipu import _SLOT_PLANS\n",
        "def f():\n    from ipu_emu.ipu import (\n        Ipu,\n        _LR_PLANS,\n    )\n",
        "def f(ipu):\n    return ipu._snapshot_buffer\n",
        "from ipu_emu import ipu\nx = ipu._INSTRUCTION_FIELD_MAP\n",
        "from ipu_emu.ipu import Ipu\nIpu._build_partition_vector(4)\n",
        "def f(ipu):\n    ipu._effective_mult_mask(0, 0, 15)\n",
    ],
)
def test_a_planted_private_use_is_reported(planted):
    assert list(engine_private_uses(planted))


@pytest.mark.parametrize(
    "allowed",
    [
        "from ipu_emu.ipu import Ipu, plans_for\n",
        "def f(ipu):\n    return ipu.state, ipu.__class__\n",
        "def f(self):\n    return self._ipu.issuer\n",
    ],
)
def test_public_uses_pass(allowed):
    assert list(engine_private_uses(allowed)) == []
