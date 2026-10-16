"""The typed ISA view and its loader (E03-09)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from ipu_isa.instruction_spec import INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.schema import SpecError
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.tags import Tag
from ipu_isa.view import (
    Instruction,
    Isa,
    Read,
    find_pseudo_instruction,
    get_instruction,
    get_instruction_by_opcode,
    get_opcode_for_instruction,
    get_operand_names_and_types,
    isa,
    load_isa,
)

SNAPSHOT = Path(__file__).parent / "fixtures" / "spec_snapshot.json"


def _operands(operands):
    return [
        {"name": op.name, "type": op.type.name, "read": op.read.value if op.read else None}
        for op in operands
    ]


def _records(view: Isa) -> dict:
    return {
        "instructions": [
            {
                "slot": i.slot.value,
                "mnemonic": i.mnemonic,
                "opcode": i.opcode,
                "operands": _operands(i.operands),
                "tag": i.tag.value,
                "execute_fn": i.execute_fn,
            }
            for slot in view.slots
            for i in view.instructions(slot)
        ],
        "pseudo_instructions": [
            {
                "mnemonic": p.mnemonic,
                "operands": _operands(p.operands),
                "expands_to": {"slot": p.expands_to.slot.value, "mnemonic": p.expands_to.mnemonic},
                "args": list(p.args),
                "tag": p.tag.value,
            }
            for p in view.pseudos()
        ],
    }


def _sort(records: dict) -> dict:
    """The snapshot lists slots in spec order; the view in registry order."""
    return {
        "instructions": sorted(records["instructions"], key=lambda r: (r["slot"], r["opcode"])),
        "pseudo_instructions": records["pseudo_instructions"],
    }


def _load(spec=INSTRUCTION_SPEC, pseudo=PSEUDO_INSTRUCTION_SPEC) -> Isa:
    return load_isa(spec, pseudo, SLOTS, OPERAND_TYPES)


def test_the_view_reproduces_the_snapshot():
    assert _sort(_records(isa())) == _sort(json.loads(SNAPSHOT.read_text()))


def test_isa_is_cached_and_equals_a_fresh_load():
    assert isa() is isa()
    assert _records(_load()) == _records(isa())


def test_lookups():
    view = isa()
    add = view.instruction(Slot.LR, "add")
    assert add.mnemonic == "ADD" and add.slot is Slot.LR and add.tag is Tag.HARDWARE
    assert view.by_opcode(Slot.LR, add.opcode) is add
    assert view.instructions(Slot.LR)[add.opcode] is add
    assert add.syntax == "ADD " + " ".join(op.name for op in add.operands)
    assert add.operand(add.operands[0].name) is add.operands[0]
    with pytest.raises(KeyError):
        add.operand("nope")
    with pytest.raises(KeyError):
        view.instruction(Slot.LR, "NOT_AN_INSTRUCTION")
    with pytest.raises(KeyError):
        view.by_opcode(Slot.LR, len(view.instructions(Slot.LR)))
    assert view.find_slot("str_acc_reg") is Slot.ACC_STORE
    with pytest.raises(ValueError, match="several slots"):
        view.find_slot("NOP")
    with pytest.raises(KeyError):
        view.find_slot("NOT_AN_INSTRUCTION")
    assert view.layout.word_bits() == 186
    assert view.slots[Slot.ACC_STORE].tag is Tag.EMULATOR


def test_pseudo_lookup_is_by_name_and_arity():
    b = isa().pseudo("b", 1)
    assert b is not None and b.tag is Tag.ASSEMBLER and b.expands_to.mnemonic == "BEQ"
    assert isa().pseudo("B", 2) is None
    assert isa().pseudo("BZ", 3) is None  # the 3-operand BZ is a real instruction


def test_read_sources():
    reads = {
        op.read for slot in Slot for i in isa().instructions(slot) for op in i.operands if op.read
    }
    assert reads <= set(Read)
    assert Read("snapshot") is Read.CYCLE_START


def test_records_are_immutable():
    view = isa()
    add = view.instruction(Slot.LR, "ADD")
    with pytest.raises(dataclasses.FrozenInstanceError):
        add.opcode = 3  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        add.operands[0].name = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        add.doc.title = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.layout = view.layout  # type: ignore[misc]
    assert isinstance(add.operands, tuple) and isinstance(add.doc.operands, tuple)
    for mapping in (view.slots, view.operand_types, view.layout.unions, view._instructions):
        with pytest.raises(TypeError):
            mapping[Slot.LR] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        view.layout.operand_bits["LrIdx"] = 1  # type: ignore[index]


def test_a_planted_schema_violation_is_rejected():
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    spec["lr"]["ADD"]["operands"][0]["raed"] = "snapshot"
    with pytest.raises(SpecError, match=r"lr\.ADD\.operands"):
        _load(spec)


def test_load_isa_checks_against_the_registries_it_is_given():
    """The slot and operand-type registries are arguments too (E03-24)."""
    without_lr_idx = {name: record for name, record in OPERAND_TYPES.items() if name != "LrIdx"}
    with pytest.raises(SpecError, match="unknown operand type 'LrIdx'"):
        load_isa(INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC, SLOTS, without_lr_idx)

    unreadable = dict(OPERAND_TYPES)
    unreadable["LrIdx"] = dataclasses.replace(OPERAND_TYPES["LrIdx"], readable=False)
    with pytest.raises(SpecError, match=r"\.read: a LrIdx operand is not a register"):
        load_isa(INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC, SLOTS, unreadable)

    spec = copy.deepcopy(INSTRUCTION_SPEC)
    spec["mult"]["NOP"]["tag"] = Tag.EMULATOR
    emulator_mult = dict(SLOTS)
    emulator_mult[Slot.MULT] = dataclasses.replace(SLOTS[Slot.MULT], tag=Tag.EMULATOR)
    order = tuple(s for s in isa().layout.layout_order if s is not Slot.MULT) + (Slot.MULT,)
    view = load_isa(spec, PSEUDO_INSTRUCTION_SPEC, emulator_mult, OPERAND_TYPES, order)
    assert view.instruction(Slot.MULT, "NOP").tag is Tag.EMULATOR
    assert {i.tag for i in view.instructions(Slot.MULT)} == {Tag.EMULATOR}
    assert view.layout.word_bits(Tag.HARDWARE) < isa().layout.word_bits(Tag.HARDWARE)


def test_a_modified_spec_loads_in_process():
    """One extra instruction appears in the view, with the next opcode (E03-24)."""
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    entry = copy.deepcopy(spec["lr"]["ADD"])
    entry["doc"] = dataclasses.replace(entry["doc"], example="ADD2 LR0 LR1 LR2;;")
    spec["lr"]["ADD2"] = entry
    view = _load(spec)
    extra = view.instruction(Slot.LR, "ADD2")
    assert isinstance(extra, Instruction)
    assert extra.opcode == len(INSTRUCTION_SPEC["lr"])
    assert len(view.instructions(Slot.LR)) == len(isa().instructions(Slot.LR)) + 1
    with pytest.raises(KeyError):
        isa().instruction(Slot.LR, "ADD2")


def test_old_helpers_keep_their_return_shapes():
    name, entry = get_instruction_by_opcode("lr", 0)
    assert name == next(iter(INSTRUCTION_SPEC["lr"]))
    original = INSTRUCTION_SPEC["lr"][name]
    assert entry == {
        "operands": original["operands"],
        "doc": original["doc"],
        "execute_fn": original["execute_fn"],
    }
    assert get_instruction("lr", name.lower()) == entry
    assert get_opcode_for_instruction("lr", name) == 0
    assert get_operand_names_and_types("lr", name) == [
        (op["name"], op["type"]) for op in original["operands"]
    ]
    pseudo = find_pseudo_instruction("bz", 2)
    assert pseudo is not None
    assert pseudo["expands_to"] == PSEUDO_INSTRUCTION_SPEC["BZ"]["expands_to"]
    assert pseudo["operands"] == PSEUDO_INSTRUCTION_SPEC["BZ"]["operands"]
    assert find_pseudo_instruction("BZ", 3) is None
