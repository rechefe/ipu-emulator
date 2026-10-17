"""One handler module per slot (E04-14): where handlers live and how they bind.

Every spec instruction's ``execute_fn`` is a function of its slot's module in
``ipu_emu.slots``, taking the engine and then, keyword-only, exactly the
instruction's operand names. ``Ipu`` holds no handler, no slot module imports
another, and no file of the package is over 800 lines.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

import ipu_emu
from ipu_emu import slots
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu import Ipu, bind_handler
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.view import isa

PACKAGE_DIR = Path(inspect.getsourcefile(ipu_emu)).parent
SLOT_MODULE_NAMES = frozenset(
    module.__name__.rsplit(".", 1)[1] for module in slots.HANDLER_MODULES.values()
)

ALL_INSTRUCTIONS = [instruction for slot in SLOTS for instruction in isa().instructions(slot)]


def test_every_slot_has_a_handler_module():
    assert set(slots.HANDLER_MODULES) == set(SLOTS)


@pytest.mark.parametrize("instruction", ALL_INSTRUCTIONS, ids=lambda i: f"{i.slot}/{i.mnemonic}")
def test_each_handler_takes_the_engine_then_its_operand_names_keyword_only(instruction):
    fn = getattr(slots.HANDLER_MODULES[instruction.slot], instruction.execute_fn)
    params = list(inspect.signature(fn).parameters.values())
    assert [p.kind for p in params[:1]] == [inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[1:])
    assert {p.name for p in params[1:]} == {op.name for op in instruction.operands}
    assert bind_handler(instruction) is fn


def test_a_handler_whose_parameters_are_not_the_operands_is_rejected(monkeypatch):
    instruction = isa().instruction(Slot.LR, "INC")

    def execute_lr_inc(ipu, *, dest, immediate):  # "imm" in the spec
        raise AssertionError("never called")

    monkeypatch.setattr(slots.HANDLER_MODULES[Slot.LR], "execute_lr_inc", execute_lr_inc)
    with pytest.raises(ConfigurationError, match=r"lr/INC names handler .*'imm'"):
        bind_handler(instruction)


def test_a_missing_handler_names_the_slot_module():
    fake = dataclasses.replace(
        isa().instruction(Slot.BREAK, "BREAK"), mnemonic="FAKE", execute_fn="execute_nothing"
    )
    with pytest.raises(ConfigurationError, match=r"ipu_emu\.slots\.brk\.execute_nothing"):
        bind_handler(fake)


def test_ipu_holds_no_handler():
    handlers = {name for name in vars(Ipu) if name.startswith("execute_")}
    assert handlers == set()


def _imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize("name", sorted(SLOT_MODULE_NAMES))
def test_no_slot_module_imports_another(name):
    imported = _imported_modules(PACKAGE_DIR / "slots" / f"{name}.py")
    others = {f"ipu_emu.slots.{other}" for other in SLOT_MODULE_NAMES - {name}}
    assert imported & (others | {"ipu_emu.slots"}) == set()


def test_every_module_is_under_800_lines():
    sizes = {
        str(path.relative_to(PACKAGE_DIR)): len(path.read_text(encoding="utf-8").splitlines())
        for path in PACKAGE_DIR.rglob("*.py")
    }
    assert {path: n for path, n in sizes.items() if n >= 800} == {}
