"""The derived binary layout of the ISA, computed once (E03-07, E03-08).

:func:`compute_layout` runs the union solver over a spec and returns an
immutable :class:`IsaLayout`: every slot's packed fields, and the width of
every operand type, including the widths only packing decides
(``LrIncDecImmediate``, ``LrOrReshapeMaskImmediate``). :func:`isa_layout` is
that value for the default spec. Nothing is written into other modules, so the
result does not depend on import order.

This module imports the spec, never the other way round.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ipu_isa.instruction_spec import INSTRUCTION_SPEC
from ipu_isa.operand_types import OPERAND_TYPES, OperandType
from ipu_isa.schema import SpecError
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot, SlotInfo
from ipu_isa.tags import Tag
from ipu_isa.union_layout import SlotUnion, compute_slot_layout


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
    spec: Mapping[str, Mapping],
    operand_types: Mapping[str, OperandType],
    slots: Mapping[Slot, SlotInfo],
    layout_order: tuple[Slot, ...] = LAYOUT_ORDER,
) -> IsaLayout:
    """Solve the layout of ``spec``. Pure: equal inputs give equal results."""
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


@functools.cache
def isa_layout() -> IsaLayout:
    """The layout of the default spec."""
    return compute_layout(INSTRUCTION_SPEC, OPERAND_TYPES, SLOTS)


def slots_with_tag(tag: Tag) -> tuple[Slot, ...]:
    """The default spec's slots tagged ``tag``, in layout order."""
    return isa_layout().slots_with_tag(tag)


def word_bits(tag: Tag | None = None) -> int:
    """Width of the default emulator word (``None``) or of its ``tag`` slots."""
    return isa_layout().word_bits(tag)


#: Slot name to its union layout, in the spec's slot order.
SLOT_UNIONS: Mapping[str, SlotUnion] = MappingProxyType(
    {slot.value: su for slot, su in isa_layout().unions.items()}
)

#: Slot name to the canonical operand type of each union field, in field order.
SLOT_BINARY_LAYOUT: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {slot: tuple(f.canonical_type for f in su.fields) for slot, su in SLOT_UNIONS.items()}
)
