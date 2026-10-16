"""The operand-type registry (E03-05)."""

from __future__ import annotations

import pytest

from ipu_isa.instruction_spec import INSTRUCTION_SPEC, VALID_OPERAND_TYPES
from ipu_isa.operand_types import OPERAND_TYPES, OperandKind, operand_type
from ipu_isa.union_layout import get_operand_type_bits


def test_every_type_has_bits_or_is_a_derived_immediate():
    for record in OPERAND_TYPES.values():
        assert record.bits is not None or record.kind is OperandKind.IMMEDIATE, record.name
    assert {r.name for r in OPERAND_TYPES.values() if r.bits is None} == {
        "LrIncDecImmediate",
        "LrOrReshapeMaskImmediate",
    }


def test_sv_typedef_is_set_exactly_for_registers_and_enums():
    for record in OPERAND_TYPES.values():
        has_typedef = record.kind in (OperandKind.REGISTER, OperandKind.ENUM)
        assert (record.sv_typedef is not None) == has_typedef, record.name


def test_field_suffixes_are_unique():
    suffixes = [record.field_suffix for record in OPERAND_TYPES.values()]
    assert len(suffixes) == len(set(suffixes))


def test_readable_types_are_the_register_operands_with_a_read_flag():
    assert {r.name for r in OPERAND_TYPES.values() if r.readable} == {
        "LrIdx",
        "CrIdx",
        "LcrIdx",
        "MultStageReg",
        "LrdIdx",
    }


def test_every_spec_operand_type_is_registered():
    used = {
        op["type"]
        for slot in INSTRUCTION_SPEC.values()
        for d in slot.values()
        for op in d["operands"]
    }
    assert used <= set(OPERAND_TYPES)
    assert VALID_OPERAND_TYPES == frozenset(OPERAND_TYPES)


def test_widths_are_what_the_union_layout_packs():
    bits = get_operand_type_bits()
    assert bits == {name: record.bits or 0 for name, record in OPERAND_TYPES.items()}
    assert bits["Label"] == 10 and bits["LcrIdx"] == 5 and bits["MultStageReg"] == 2


def test_enum_and_register_widths_fit_their_domains():
    for record in OPERAND_TYPES.values():
        if record.kind in (OperandKind.REGISTER, OperandKind.ENUM):
            assert len(record.domain.members) <= 1 << record.bits, record.name


def test_unknown_type_names_the_valid_ones():
    with pytest.raises(KeyError, match="known: .*LrIdx"):
        operand_type("LrIndex")


def test_registry_is_read_only():
    with pytest.raises(TypeError):
        OPERAND_TYPES["LrIdx"] = OPERAND_TYPES["CrIdx"]
    with pytest.raises(AttributeError):
        OPERAND_TYPES["LrIdx"].bits = 5
