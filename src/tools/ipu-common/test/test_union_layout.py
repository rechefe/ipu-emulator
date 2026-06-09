"""Tests for the union layout solver."""
from __future__ import annotations

from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    SLOT_METADATA,
    SLOT_UNIONS,
    is_hardware_slot,
)
from ipu_common.union_layout import compute_slot_layouts, get_operand_type_bits


def test_solver_is_deterministic():
    """Running the solver twice produces identical results."""
    type_bits = get_operand_type_bits()
    first = compute_slot_layouts(INSTRUCTION_SPEC)
    second = compute_slot_layouts(INSTRUCTION_SPEC)
    for slot in first:
        su1, su2 = first[slot], second[slot]
        assert su1.opcode_bits == su2.opcode_bits
        assert len(su1.fields) == len(su2.fields)
        for f1, f2 in zip(su1.fields, su2.fields):
            assert f1.canonical_type == f2.canonical_type
            assert f1.bits == f2.bits


def test_no_co_occurring_operands_share_a_field():
    """No two operands that appear in the same instruction map to the same union field."""
    for slot, su in SLOT_UNIONS.items():
        for opcode, bindings in su.opcode_bindings.items():
            field_indices = [fi for fi, _ in bindings]
            assert len(field_indices) == len(set(field_indices)), (
                f"{slot}.{opcode}: operands share a field index: {bindings}"
            )


def test_all_opcodes_have_bindings():
    """Every opcode in the instruction spec appears in opcode_bindings."""
    for slot, instructions in INSTRUCTION_SPEC.items():
        su = SLOT_UNIONS[slot]
        for opcode in instructions:
            assert opcode in su.opcode_bindings, (
                f"{slot}.{opcode} missing from opcode_bindings"
            )


def test_field_count_not_greater_than_max_operands():
    """Union field count ≤ max operands in any single instruction (bin-packing saves fields)."""
    for slot, instructions in INSTRUCTION_SPEC.items():
        su = SLOT_UNIONS[slot]
        max_ops = max((len(d["operands"]) for d in instructions.values()), default=0)
        assert len(su.fields) <= max_ops, (
            f"{slot}: {len(su.fields)} fields but max operands is {max_ops}"
        )


def test_canonical_type_is_widest_in_bin():
    """Each union field's canonical type has the maximum bit-width among all types in that bin."""
    type_bits = get_operand_type_bits()
    for slot, su in SLOT_UNIONS.items():
        for f in su.fields:
            canonical_bits = type_bits[f.canonical_type]
            for opcode, (operand_name, actual_type) in f.users.items():
                assert type_bits[actual_type] <= canonical_bits, (
                    f"{slot} field[{f.index}]: user {opcode}.{operand_name} has type "
                    f"{actual_type} ({type_bits[actual_type]} bits) > canonical "
                    f"{f.canonical_type} ({canonical_bits} bits)"
                )


def test_lr_slot_sharing():
    """The LR slot uses exactly 3 fields (down from 6 in a naive struct layout)."""
    su = SLOT_UNIONS["lr"]
    assert len(su.fields) == 3, (
        f"Expected 3 union fields for LR slot, got {len(su.fields)}"
    )


def test_all_slots_present():
    """All expected slots are computed."""
    expected = {"load", "store", "acc_store", "lr", "mult", "acc", "aaq", "cond", "break"}
    assert set(SLOT_UNIONS.keys()) == expected


def test_acc_store_slot_marked_non_hardware():
    """The acc_store slot is flagged simulation-only in SLOT_METADATA."""
    assert SLOT_METADATA["acc_store"]["hardware"] is False
    assert is_hardware_slot("load")
    assert is_hardware_slot("store")
    assert not is_hardware_slot("acc_store")
