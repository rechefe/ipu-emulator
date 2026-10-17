"""Read-only stage descriptions of an instruction word, from the decoded word and the slot registry.

Each slot instance is decoded on its own by the ``ipu_isa`` slot codec; its
label and the stage it belongs to come from the slot metadata, so another LR
sub-slot, or another slot, reaches the view with no edit here.
"""

from __future__ import annotations

from collections.abc import Mapping

from ipu_isa.codec.disasm import operation_text
from ipu_isa.codec.slot import SLOT_CODECS
from ipu_isa.codec.word import DecodedWord
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot
from ipu_isa.timing import CONTROL_SLOTS, SLOT_EXECUTION_ORDER

#: Every slot instance of a word, most significant first.
SLOT_INSTANCES: tuple[tuple[Slot, int], ...] = tuple(
    (slot, instance) for slot in LAYOUT_ORDER for instance in range(SLOTS[slot].count)
)


def _label(slot: Slot, instance: int) -> str:
    """``LR1``, ``MULT``; ``*`` marks a debug-only slot."""
    return SLOTS[slot].instance_labels()[instance] + ("*" if SLOTS[slot].debug_only else "")


#: One label per slot instance, in layout order; ``*`` marks debug-only slots.
SLOT_LABELS = tuple(_label(slot, instance) for slot, instance in SLOT_INSTANCES)


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


def _stages() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The stage view's grouping, from the ISA's slot data alone (E08-07).

    CTRL is the timing contract's ``CONTROL_SLOTS``. Every other slot is a
    stage of its own, in ``SLOT_EXECUTION_ORDER`` (a slot that order does not
    list comes after it), and the debug-only slots are grouped last as
    Simulation. A new slot therefore appears with no edit here.
    """
    ordered = [*SLOT_EXECUTION_ORDER, *(s for s in LAYOUT_ORDER if s not in SLOT_EXECUTION_ORDER)]
    control = [s for s in LAYOUT_ORDER if s in CONTROL_SLOTS]
    rest = [s for s in ordered if s not in CONTROL_SLOTS]
    stages = [_stage("CTRL", control)]
    stages += [_stage(str(s).upper(), (s,)) for s in rest if not SLOTS[s].debug_only]
    simulation = [s for s in rest if SLOTS[s].debug_only]
    if simulation:
        stages.append(_stage("Simulation", simulation))
    return tuple(stages)


#: How the stage view groups the slots (see :func:`_stages`).
PIPELINE_STAGES = _stages()


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
