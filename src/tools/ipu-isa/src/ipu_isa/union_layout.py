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
7. A slot with ``pad_to_bits`` grows one field to reach that width.

The solver is pure: it takes the bit widths and padding targets as arguments
and returns new frozen values. :func:`compute_layout` runs it over a whole
spec into an :class:`IsaLayout`; ``ipu_isa.view.load_isa`` calls it once for
the repository's ISA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ipu_isa.operand_types import OPERAND_TYPES, OperandType
from ipu_isa.schema import SpecError
from ipu_isa.slots import LAYOUT_ORDER, Slot, SlotInfo
from ipu_isa.tags import Tag


def get_operand_type_bits() -> dict[str, int]:
    """Return the authored bit-width of each operand type in the registry.

    A width decided by union packing is 0 here; ``ipu_isa.layout`` reads the
    packed width from the field that carries the type.
    """
    return {name: operand_type.bits or 0 for name, operand_type in OPERAND_TYPES.items()}


@dataclass(frozen=True)
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
    users: Mapping[str, tuple[str, str]]


@dataclass(frozen=True)
class SlotUnion:
    """Complete union layout for one VLIW slot.

    Attributes:
        slot: Slot name (e.g. ``"lr"``, ``"mult"``).
        opcode_bits: Number of bits used by the opcode field.
        fields: Ordered union fields (opcode not included).
        opcode_bindings: Maps opcode_name → ((field_index, operand_name), ...)
            in operand declaration order.  Every opcode appears here; those
            with no operands map to an empty tuple.
    """

    slot: str
    opcode_bits: int
    fields: tuple[UnionField, ...]
    opcode_bindings: Mapping[str, tuple[tuple[int, str], ...]]

    @property
    def bits(self) -> int:
        """Width of one instance of the slot: opcode plus every field."""
        return self.opcode_bits + sum(f.bits for f in self.fields)


def compute_slot_layout(
    slot_name: str,
    instructions: Mapping[str, Mapping[str, Any]],
    type_bits: Mapping[str, int],
    pad_to_bits: int | None = None,
) -> SlotUnion:
    """Compute the union layout for a single slot, padded to ``pad_to_bits``."""

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
        (t, si): bin_idx for bin_idx, bin_ in enumerate(bins) for (t, si, _) in bin_
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
        for t, si, user_set in bin_:
            for opcode_name in user_set:
                count = 0
                for op in instructions[opcode_name]["operands"]:
                    if op["type"] == t:
                        if count == si:
                            users_map[opcode_name] = (op["name"], t)
                            break
                        count += 1

        fields.append(
            UnionField(
                index=bin_idx,
                canonical_type=canonical,
                bits=max_width,
                users=MappingProxyType(users_map),
            )
        )

    # ------------------------------------------------------------------ #
    # Step 7: build opcode_bindings                                       #
    # ------------------------------------------------------------------ #
    opcode_bindings: dict[str, tuple[tuple[int, str], ...]] = {
        opcode_name: tuple(
            (slot_to_bin[(t, si)], op["name"])
            for op, (t, si) in zip(
                instructions[opcode_name]["operands"],
                slot_assignment[opcode_name],
                strict=False,
            )
        )
        for opcode_name in instructions
    }

    n_opcodes = len(instructions)
    opcode_bits = max(1, (n_opcodes - 1).bit_length()) if n_opcodes > 1 else 1

    return _pad_slot(
        SlotUnion(
            slot=slot_name,
            opcode_bits=opcode_bits,
            fields=tuple(fields),
            opcode_bindings=MappingProxyType(opcode_bindings),
        ),
        pad_to_bits,
    )


def _pad_slot(su: SlotUnion, target: int | None) -> SlotUnion:
    """Grow one field when packing is narrower than the slot's wire width."""
    if target is None:
        return su
    deficit = target - su.bits
    if deficit <= 0:
        return su
    # Prefer padding the field that carries LrModPow2K (k); the extra bit is unused.
    for slot_field in su.fields:
        if any(
            actual_type == "LrModPow2KImmediate"
            for _opcode, (_operand_name, actual_type) in slot_field.users.items()
        ):
            padded = replace(slot_field, bits=slot_field.bits + deficit)
            fields = tuple(padded if f is slot_field else f for f in su.fields)
            return replace(su, fields=fields)
    raise ValueError(f"Cannot pad {su.slot} slot to {target} bits (currently {su.bits})")


@dataclass(frozen=True)
class IsaLayout:
    """The packed layout of every slot and the width of every operand type."""

    #: Slot to its union layout, in the spec's slot order.
    unions: Mapping[Slot, SlotUnion]
    #: Operand type name to its field width, derived widths filled in.
    operand_bits: Mapping[str, int]
    #: Each slot's tag and instance count, as the word is built from them.
    slots: Mapping[Slot, SlotInfo]
    #: Slots in the word, most significant first.
    layout_order: tuple[Slot, ...]

    def slot_width(self, slot: Slot) -> int:
        """Bits of one instance of ``slot``."""
        return self.unions[slot].bits

    def slots_with_tag(self, tag: Tag) -> tuple[Slot, ...]:
        """The slots tagged ``tag``, in layout order."""
        return tuple(slot for slot in self.layout_order if self.slots[slot].tag is tag)

    def word_bits(self, tag: Tag | None = None) -> int:
        """Width of the emulator word (``None``) or of its ``tag`` slots only.

        ``word_bits(Tag.HARDWARE)`` is the hardware word: the most significant
        bits of the emulator word (Q41).
        """
        slots = self.layout_order if tag is None else self.slots_with_tag(tag)
        return sum(self.slot_width(slot) * self.slots[slot].count for slot in slots)


def _derived_operand_bits(
    unions: Mapping[Slot, SlotUnion], operand_types: Mapping[str, OperandType]
) -> dict[str, int]:
    """Every operand type's width; a type without an authored width takes the
    width of the field that carries it, which must be the same everywhere."""
    bits = {name: record.bits for name, record in operand_types.items() if record.bits}
    packed: dict[str, set[int]] = {}
    for su in unions.values():
        for field in su.fields:
            for _operand_name, actual_type in field.users.values():
                packed.setdefault(actual_type, set()).add(field.bits)
    for name, record in operand_types.items():
        if record.bits:
            continue
        widths = packed.get(name)
        if not widths:
            raise SpecError(f"OPERAND_TYPES.{name}: width is derived but no instruction uses it")
        if len(widths) != 1:
            raise SpecError(
                f"OPERAND_TYPES.{name}: width is derived but its fields differ: {sorted(widths)}"
            )
        (bits[name],) = widths
    return bits


def compute_layout(
    spec: Mapping[str, Mapping[str, Any]],
    operand_types: Mapping[str, OperandType],
    slots: Mapping[Slot, SlotInfo],
    layout_order: tuple[Slot, ...] = LAYOUT_ORDER,
) -> IsaLayout:
    """Solve the layout of ``spec``. Pure: equal inputs give equal results."""
    if set(spec) != set(slots) or sorted(layout_order) != sorted(slots):
        raise SpecError(
            f"compute_layout: spec slots {sorted(spec)}, slot registry {sorted(slots)} and "
            f"layout order {sorted(layout_order)} must name the same slots"
        )
    type_bits = {name: record.bits or 0 for name, record in operand_types.items()}
    unions = {
        Slot(slot): compute_slot_layout(
            slot, instructions, type_bits, slots[Slot(slot)].pad_to_bits
        )
        for slot, instructions in spec.items()
    }
    return IsaLayout(
        unions=MappingProxyType(unions),
        operand_bits=MappingProxyType(_derived_operand_bits(unions, operand_types)),
        slots=MappingProxyType(dict(slots)),
        layout_order=tuple(layout_order),
    )
