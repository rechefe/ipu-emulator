"""HARDWARE / EMULATOR / ASSEMBLER tags and the hardware word (E03-07)."""

from __future__ import annotations

import copy

import pytest

from ipu_isa.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    SLOT_METADATA,
    validate_instruction_spec,
    validate_pseudo_instruction_spec,
)
from ipu_isa.layout import slots_with_tag, word_bits
from ipu_isa.schema import SpecError
from ipu_isa.slots import _SPEC, LAYOUT_ORDER, SLOTS, Slot, _load
from ipu_isa.tags import Tag
from ipu_isa.view import isa


def test_word_widths():
    assert word_bits() == 186
    assert word_bits(Tag.HARDWARE) == 155
    assert word_bits(Tag.EMULATOR) == 186 - 155
    assert word_bits(Tag.ASSEMBLER) == 0


def test_slot_tags():
    assert slots_with_tag(Tag.EMULATOR) == (Slot.ACC_STORE, Slot.BREAK)
    assert [slot for slot, info in SLOTS.items() if info.debug_only] == [Slot.ACC_STORE]


def test_hardware_slots_are_a_prefix_of_the_layout_order():
    hardware = slots_with_tag(Tag.HARDWARE)
    assert LAYOUT_ORDER[: len(hardware)] == hardware


def test_every_instruction_and_pseudo_has_one_tag():
    for slot in Slot:
        for instruction in isa().instructions(slot):
            assert instruction.tag is SLOTS[slot].tag
    assert {pseudo.tag for pseudo in isa().pseudos()} == {Tag.ASSEMBLER}


def _with(slot, field, value):
    spec = {s: dict(entry) for s, entry in _SPEC.items()}
    spec[slot][field] = value
    return spec


def test_an_emulator_slot_before_a_hardware_slot_is_rejected():
    order = list(LAYOUT_ORDER)
    order.remove(Slot.ACC_STORE)
    order.insert(order.index(Slot.STORE), Slot.ACC_STORE)
    with pytest.raises(SpecError, match="HARDWARE slot 'store' follows an EMULATOR slot"):
        _load(_SPEC, tuple(order))


def test_a_slot_cannot_be_assembler():
    with pytest.raises(SpecError, match=r"SLOTS\.mult\.tag"):
        _load(_with(Slot.MULT, "tag", Tag.ASSEMBLER), LAYOUT_ORDER)


def test_a_slot_tag_must_be_a_tag():
    with pytest.raises(SpecError, match=r"SLOTS\.mult\.tag"):
        _load(_with(Slot.MULT, "tag", "hardware"), LAYOUT_ORDER)


def test_a_debug_only_slot_must_be_emulator():
    with pytest.raises(SpecError, match=r"SLOTS\.mult\.debug_only"):
        _load(_with(Slot.MULT, "debug_only", True), LAYOUT_ORDER)


@pytest.mark.parametrize(
    "slot, mnemonic, tag",
    [
        ("acc_store", "ST.ACC", Tag.HARDWARE),  # an EMULATOR slot holds EMULATOR instructions
        ("lr", "ADD", Tag.EMULATOR),  # no EMULATOR instruction in a HARDWARE slot (yet)
        ("lr", "ADD", Tag.ASSEMBLER),
        ("lr", "ADD", "hardware"),
    ],
)
def test_an_instruction_has_its_slot_tag(slot, mnemonic, tag):
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    spec[slot][mnemonic]["tag"] = tag
    with pytest.raises(SpecError, match=rf"{slot}\.{mnemonic}\.tag"):
        validate_instruction_spec(spec, SLOT_METADATA)


def test_an_instruction_may_state_its_slot_tag():
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    spec["acc_store"]["ST.ACC"]["tag"] = Tag.EMULATOR
    validate_instruction_spec(spec, SLOT_METADATA)


def test_a_pseudo_tagged_hardware_is_rejected():
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    name = next(iter(pseudo))
    pseudo[name]["tag"] = Tag.HARDWARE
    with pytest.raises(SpecError, match=rf"pseudo\.{name}\.tag"):
        validate_pseudo_instruction_spec(pseudo, INSTRUCTION_SPEC)


def test_a_pseudo_must_expand_to_a_real_instruction():
    """A target tagged ASSEMBLER (a planted, invalid spec) is rejected."""
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    pseudo = copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)
    name, pseudo_def = next(iter(pseudo.items()))
    target = pseudo_def["expands_to"]
    real = next(n for n in spec[target["slot"]] if n.upper() == target["instruction"].upper())
    spec[target["slot"]][real]["tag"] = Tag.ASSEMBLER
    with pytest.raises(SpecError, match=rf"pseudo\.{name}\.expands_to\.instruction"):
        validate_pseudo_instruction_spec(pseudo, spec)
