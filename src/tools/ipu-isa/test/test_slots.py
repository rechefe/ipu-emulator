"""The slot registry (E03-06)."""

from __future__ import annotations

import pytest

from ipu_isa.instruction_spec import (
    COMPOUND_LAYOUT_SLOT_ORDER,
    INSTRUCTION_SPEC,
    SLOT_COUNT,
    SLOT_METADATA,
)
from ipu_isa.schema import SpecError
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot, _load, _SPEC


def test_layout_order_covers_every_slot_once():
    assert sorted(LAYOUT_ORDER) == sorted(Slot)
    assert len(LAYOUT_ORDER) == len(set(LAYOUT_ORDER))


def test_instance_labels():
    assert SLOTS[Slot.LR].instance_labels() == ("LR0", "LR1", "LR2")
    assert SLOTS[Slot.MULT].instance_labels() == ("MULT",)
    assert SLOTS[Slot.LR].field_prefixes() == ("lr_inst_0", "lr_inst_1", "lr_inst_2")


@pytest.mark.parametrize("key", ["field_prefix", "opcode_enum", "sv_struct_base", "asm_opcode_class"])
def test_wire_names_are_unique(key):
    values = [getattr(info, key) for info in SLOTS.values()]
    assert len(values) == len(set(values))


def test_derived_views_equal_the_old_literals():
    assert SLOT_COUNT == {
        "break": 1, "load": 1, "store": 1, "acc_store": 1, "mult": 1,
        "acc": 1, "aaq": 1, "lr": 3, "cond": 1,
    }
    assert COMPOUND_LAYOUT_SLOT_ORDER == [
        "cond", "lr", "load", "mult", "acc", "aaq", "store", "acc_store", "break",
    ]
    assert SLOT_METADATA["acc_store"] == {"description": SLOTS[Slot.ACC_STORE].description}


def test_slot_is_its_string():
    assert Slot.LR == "lr" and str(Slot.LR) == "lr" and f"{Slot.ACC_STORE}" == "acc_store"
    assert set(INSTRUCTION_SPEC) == {slot.value for slot in Slot}


def test_layout_order_must_list_every_slot_once():
    with pytest.raises(SpecError, match="LAYOUT_ORDER"):
        _load(_SPEC, LAYOUT_ORDER[:-1] + (Slot.COND,))


def test_entries_are_key_checked():
    spec = {slot: dict(entry) for slot, entry in _SPEC.items()}
    spec[Slot.MULT]["colour"] = "#000000"
    with pytest.raises(SpecError, match=r"SLOTS\.mult: unknown key\(s\) 'colour'"):
        _load(spec, LAYOUT_ORDER)


def test_every_slot_needs_an_entry():
    spec = {slot: entry for slot, entry in _SPEC.items() if slot is not Slot.AAQ}
    with pytest.raises(SpecError, match="SLOTS: entries .* do not match Slot"):
        _load(spec, LAYOUT_ORDER)


def test_lr_is_padded_to_20_bits():
    assert SLOTS[Slot.LR].pad_to_bits == 20
    assert all(info.pad_to_bits is None for slot, info in SLOTS.items() if slot is not Slot.LR)
