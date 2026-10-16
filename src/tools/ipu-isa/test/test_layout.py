"""The derived layout: computed once, immutable, independent of import order (E03-08)."""

from __future__ import annotations

import os

import pytest

from ipu_isa.instruction_spec import INSTRUCTION_SPEC
from ipu_isa.layout import SLOT_BINARY_LAYOUT, SLOT_UNIONS, compute_layout, isa_layout
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS, Slot


def test_compute_layout_is_pure():
    first = compute_layout(INSTRUCTION_SPEC, OPERAND_TYPES, SLOTS)
    second = compute_layout(INSTRUCTION_SPEC, OPERAND_TYPES, SLOTS)
    assert first == second
    assert first == isa_layout()
    assert isa_layout() is isa_layout()


def test_layout_is_immutable():
    layout = isa_layout()
    lr = layout.unions[Slot.LR]
    with pytest.raises(TypeError):
        layout.unions[Slot.LR] = lr
    with pytest.raises(TypeError):
        layout.operand_bits["LrIncDecImmediate"] = 9
    with pytest.raises(TypeError):
        lr.opcode_bindings["NOP"] = ()
    with pytest.raises(TypeError):
        lr.fields[0].users["NOP"] = ("x", "y")
    with pytest.raises(AttributeError):
        lr.fields[0].bits = 1  # frozen dataclass: FrozenInstanceError
    assert isinstance(lr.fields, tuple)
    with pytest.raises(TypeError):
        SLOT_UNIONS["lr"] = lr


def test_derived_operand_widths():
    bits = isa_layout().operand_bits
    assert bits["LrIncDecImmediate"] == 8
    assert bits["LrOrReshapeMaskImmediate"] == 4
    assert set(bits) == set(OPERAND_TYPES)


def test_slot_views_agree_with_the_layout():
    assert set(SLOT_UNIONS) == {slot.value for slot in Slot}
    for slot, su in SLOT_UNIONS.items():
        assert SLOT_BINARY_LAYOUT[slot] == tuple(f.canonical_type for f in su.fields)
        assert isa_layout().slot_width(Slot(slot)) == su.bits
    assert isa_layout().slot_width(Slot.LR) == 20


def test_a_modified_spec_gets_its_own_layout():
    """compute_layout reads only its arguments (E03-24 relies on this)."""
    spec = {slot: dict(instructions) for slot, instructions in INSTRUCTION_SPEC.items()}
    spec["break"] = {"NOP": INSTRUCTION_SPEC["break"]["NOP"]}
    layout = compute_layout(spec, OPERAND_TYPES, SLOTS)
    assert layout.unions[Slot.BREAK].fields == ()
    assert layout.word_bits() < isa_layout().word_bits()
    assert isa_layout().unions[Slot.BREAK].fields != ()


def test_no_ipu_isa_module_sets_attributes_on_another():
    """No ``setattr(module, ...)`` and no global left as a placeholder."""
    import ipu_isa

    root = os.path.dirname(ipu_isa.__file__)
    for name in os.listdir(root):
        if name.endswith(".py"):
            with open(os.path.join(root, name)) as f:
                assert "setattr(" not in f.read(), name
