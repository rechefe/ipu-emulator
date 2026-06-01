"""Automatic union layout solver for IPU instruction slots.

Given INSTRUCTION_SPEC, this module computes the optimal binary field layout
for each slot by minimising total bit-width while correctly sharing fields
across opcodes that never co-occur.

Algorithm (per slot)
====================
1. For each operand type T, count the maximum simultaneous uses in any single
   instruction.  Allocate that many *atomic slots* for T.
2. Assign each instruction's operands to atomic slots (greedy, declaration
   order within each type).
3. Build a *user set* for every atomic slot — the set of opcode names whose
   encoding touches that slot.
4. Greedy bin-packing: merge atomic slots into *union fields* when no
   instruction uses both (disjoint user sets).  Atomic slots are processed in
   alphabetical order (type name then slot index) for determinism.
5. Canonical type for a merged field = widest type; ties broken by
   alphabetical type name (first alphabetically wins).
6. Field order = bin-creation order (deterministic by the sorted input).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ipu_common.acc_agg_enums import AGG_MODE_NAMES, POST_FN_NAMES
from ipu_common.acc_stride_enums import (
    ELEMENTS_IN_ROW_NAMES,
    HORIZONTAL_STRIDE_NAMES,
    VERTICAL_STRIDE_NAMES,
)
from ipu_common.activations import ACTIVATION_FN_NAMES
from ipu_common.incr_mod_pow2_k import LR_MOD_POW2_K_FIELD_BITS
from ipu_common.mult_mask_offset import MULT_MASK_OFFSET_FIELD_BITS
from ipu_common.registers import REGISTER_DEFINITIONS


def _enum_bits(names: tuple) -> int:
    return (len(names) - 1).bit_length()


def get_operand_type_bits() -> dict[str, int]:
    """Return the bit-width for each operand type string used in instruction_spec."""
    lr_count: int = REGISTER_DEFINITIONS["lr"]["count"]
    cr_count: int = REGISTER_DEFINITIONS["cr"]["count"]
    aaq_count: int = REGISTER_DEFINITIONS["aaq"]["count"]

    return {
        "MultStageReg": 2,  # MultStageRegField overrides bits() → 2
        "LrIdx": (lr_count - 1).bit_length(),
        "CrIdx": (cr_count - 1).bit_length(),
        "LcrIdx": (lr_count + cr_count - 1).bit_length(),
        "AaqRegIdx": (aaq_count - 1).bit_length(),
        "AddSubSrcB": 6,  # 6-bit: 0-31 register codes, 32-63 IMM5
        "LrModPow2KImmediate": LR_MOD_POW2_K_FIELD_BITS,
        "MultMaskOffsetImmediate": MULT_MASK_OFFSET_FIELD_BITS,
        "BreakImmediate": 16,
        "Label": 10,  # (MAX_PROGRAM_SIZE - 1).bit_length() for size 1024
        "ElementsInRow": _enum_bits(ELEMENTS_IN_ROW_NAMES),
        "HorizontalStride": _enum_bits(HORIZONTAL_STRIDE_NAMES),
        "VerticalStride": _enum_bits(VERTICAL_STRIDE_NAMES),
        "AggMode": _enum_bits(AGG_MODE_NAMES),
        "PostFn": _enum_bits(POST_FN_NAMES),
        "ActivationFn": _enum_bits(ACTIVATION_FN_NAMES),
        "FullXmemRow": 1,
    }


@dataclass
class UnionField:
    """One field in a slot's union layout.

    Attributes:
        index: 0-based position in the slot's non-opcode field list.
        canonical_type: Operand type name that determines bit-width and the
            field-key suffix in the decoded instruction dict.
        bits: Bit-width of this field (== type_bits[canonical_type]).
        users: Maps opcode_name → (operand_name, actual_type) for every
            instruction that writes a meaningful value here.  Instructions
            that leave this field at its default are absent from the dict.
    """
    index: int
    canonical_type: str
    bits: int
    users: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class SlotUnion:
    """Complete union layout for one VLIW slot.

    Attributes:
        slot: Slot name (e.g. ``"lr"``, ``"mult"``).
        opcode_bits: Number of bits used by the opcode field.
        fields: Ordered list of union fields (opcode not included).
        opcode_bindings: Maps opcode_name → [(field_index, operand_name)]
            in operand declaration order.  Every opcode appears here; those
            with no operands map to an empty list.
    """
    slot: str
    opcode_bits: int
    fields: list[UnionField]
    opcode_bindings: dict[str, list[tuple[int, str]]]


def compute_slot_layout(
    slot_name: str,
    instructions: dict,
    type_bits: dict[str, int],
) -> SlotUnion:
    """Compute the union layout for a single slot."""

    # ------------------------------------------------------------------ #
    # Step 1 & 2: assign each operand to a typed atomic slot              #
    # ------------------------------------------------------------------ #
    # type_slots[type_name][slot_idx] = mutable set of opcode names using it
    type_slots: dict[str, list[set[str]]] = {}
    # slot_assignment[opcode_name] = [(type_name, slot_idx), ...] per operand
    slot_assignment: dict[str, list[tuple[str, int]]] = {}

    for opcode_name, inst_def in instructions.items():
        assignment: list[tuple[str, int]] = []
        type_cursor: dict[str, int] = {}
        for op in inst_def["operands"]:
            t = op["type"]
            idx = type_cursor.get(t, 0)
            type_cursor[t] = idx + 1
            if t not in type_slots:
                type_slots[t] = []
            while len(type_slots[t]) <= idx:
                type_slots[t].append(set())
            type_slots[t][idx].add(opcode_name)
            assignment.append((t, idx))
        slot_assignment[opcode_name] = assignment

    # ------------------------------------------------------------------ #
    # Step 3: sorted list of atomic slots (alphabetical type, then idx)  #
    # ------------------------------------------------------------------ #
    atomic_slots: list[tuple[str, int, frozenset[str]]] = [
        (t, i, frozenset(users))
        for t in sorted(type_slots)
        for i, users in enumerate(type_slots[t])
    ]

    # ------------------------------------------------------------------ #
    # Step 4: greedy bin-packing                                          #
    # ------------------------------------------------------------------ #
    bins: list[list[tuple[str, int, frozenset[str]]]] = []
    for t, idx, users in atomic_slots:
        placed = False
        for bin_ in bins:
            if all(users.isdisjoint(eu) for _, _, eu in bin_):
                bin_.append((t, idx, users))
                placed = True
                break
        if not placed:
            bins.append([(t, idx, users)])

    # (t, slot_idx) → bin_index
    slot_to_bin: dict[tuple[str, int], int] = {
        (t, si): bin_idx
        for bin_idx, bin_ in enumerate(bins)
        for (t, si, _) in bin_
    }

    # ------------------------------------------------------------------ #
    # Step 5 & 6: build UnionField objects                                #
    # ------------------------------------------------------------------ #
    fields: list[UnionField] = []
    for bin_idx, bin_ in enumerate(bins):
        max_width = max(type_bits[t] for t, _, _ in bin_)
        canonical = min(
            (t for t, _, _ in bin_ if type_bits[t] == max_width),
        )

        # Build users dict: opcode_name → (operand_name, actual_type)
        users_map: dict[str, tuple[str, str]] = {}
        for (t, si, user_set) in bin_:
            for opcode_name in user_set:
                count = 0
                for op in instructions[opcode_name]["operands"]:
                    if op["type"] == t:
                        if count == si:
                            users_map[opcode_name] = (op["name"], t)
                            break
                        count += 1

        fields.append(UnionField(
            index=bin_idx,
            canonical_type=canonical,
            bits=max_width,
            users=users_map,
        ))

    # ------------------------------------------------------------------ #
    # Step 7: build opcode_bindings                                       #
    # ------------------------------------------------------------------ #
    opcode_bindings: dict[str, list[tuple[int, str]]] = {
        opcode_name: [
            (slot_to_bin[(t, si)], op["name"])
            for op, (t, si) in zip(
                instructions[opcode_name]["operands"],
                slot_assignment[opcode_name],
            )
        ]
        for opcode_name in instructions
    }

    n_opcodes = len(instructions)
    opcode_bits = max(1, (n_opcodes - 1).bit_length()) if n_opcodes > 1 else 1

    return SlotUnion(
        slot=slot_name,
        opcode_bits=opcode_bits,
        fields=fields,
        opcode_bindings=opcode_bindings,
    )


def compute_slot_layouts(instruction_spec: dict) -> dict[str, SlotUnion]:
    """Compute union layouts for all slots in *instruction_spec*."""
    type_bits = get_operand_type_bits()
    return {
        slot: compute_slot_layout(slot, instructions, type_bits)
        for slot, instructions in instruction_spec.items()
    }
