"""Read-only VLIW stage descriptions, from the decoded word and the slot registry.

Each slot instance is decoded on its own by the ``ipu_isa`` slot codec; its
label and the stage it belongs to come from the slot metadata, so another LR
sub-slot, or another slot, reaches the view with no edit here.
"""

from __future__ import annotations

from collections.abc import Mapping

from ipu_as.encode import SLOT_INSTANCES, operation_text
from ipu_isa.codec.slot import SLOT_CODECS
from ipu_isa.codec.word import DecodedWord
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.timing import CONTROL_SLOTS


def _label(slot: Slot, instance: int) -> str:
    """``LR1``, ``MULT``; ``*`` marks a debug-only slot."""
    return SLOTS[slot].instance_labels()[instance] + ("*" if SLOTS[slot].debug_only else "")


#: One label per slot instance, in layout order; ``*`` marks debug-only slots.
SLOT_LABELS = tuple(_label(slot, instance) for slot, instance in SLOT_INSTANCES)
_SLOT_OF_LABEL = {label: key[0] for label, key in zip(SLOT_LABELS, SLOT_INSTANCES, strict=True)}


def _slot_of(label: str) -> Slot:
    return _SLOT_OF_LABEL[label]


def stage_operations(instruction: Mapping[str, int] | None) -> list[tuple[str, str]]:
    """Each slot instance's label and operation (``"ADD lr0 lr1 lr2"``, ``"NOP"``).

    Args:
        instruction: A decoded word (wire field name -> value), or None for an
            empty instruction memory entry.

    Returns:
        One ``(label, operation)`` per slot instance, in layout order; operands
        are in assembly order and unused fields are left out.
    """
    if instruction is None:
        return [(label, "NOP") for label in SLOT_LABELS]
    word = instruction if isinstance(instruction, DecodedWord) else DecodedWord(instruction)
    return [
        (label, operation_text(SLOT_CODECS[slot].decode(word.slot_value(slot, instance))))
        for label, (slot, instance) in zip(SLOT_LABELS, SLOT_INSTANCES, strict=True)
    ]


def display_operation(operation: str) -> str:
    """Assembly-style casing and operand separators for the stage view."""
    mnemonic, _, operands = operation.partition(" ")
    if mnemonic.upper() == "NOP":
        return "IDLE / NOP"
    return mnemonic.upper() + (" " + ", ".join(operands.upper().split()) if operands else "")


def _stage(name: str, slots) -> tuple[str, tuple[str, ...]]:
    """A stage and the labels of every instance of its slots."""
    return name, tuple(_label(slot, i) for slot in slots for i in range(SLOTS[slot].count))


#: How the stage view groups the slots: the physical pipeline, which the ISA
#: does not describe. The Control stage is the timing contract's
#: CONTROL_SLOTS; LOAD is the XMEM sideband; debug-only slots are simulation.
#: Labels come from the slot registry, so instance counts follow the ISA.
PIPELINE_STAGES = (
    _stage("CTRL", (Slot.LR, Slot.COND, Slot.BREAK)),
    _stage("MULT", (Slot.MULT,)),
    _stage("ACC", (Slot.ACC,)),
    _stage("AAQ", (Slot.AAQ,)),
    _stage("STORE", (Slot.STORE,)),
    _stage("XMEM input", (Slot.LOAD,)),
    _stage("Simulation", tuple(slot for slot in SLOTS if SLOTS[slot].debug_only)),
)

if sorted(label for _, labels in PIPELINE_STAGES for label in labels) != sorted(SLOT_LABELS):
    raise RuntimeError("PIPELINE_STAGES must place every slot instance in exactly one stage")
if set(PIPELINE_STAGES[0][1]) != {
    label for label in SLOT_LABELS if _slot_of(label) in CONTROL_SLOTS
}:
    raise RuntimeError("the CTRL stage must be the timing contract's CONTROL_SLOTS")


def pipeline_occupancy(
    instruction: dict[str, int] | None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Commands ready in each physical stage at the paused cycle boundary.

    NOPs are empty stages, not previously completed commands held in flight.
    Several control operations may occupy CTRL concurrently.
    """
    slots = dict(stage_operations(instruction))
    return [
        (
            stage,
            [(slot, slots[slot]) for slot in members if slots[slot].split()[0].upper() != "NOP"],
        )
        for stage, members in PIPELINE_STAGES
    ]
