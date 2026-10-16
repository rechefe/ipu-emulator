"""Read-only VLIW stage descriptions derived from the ISA's disassembler."""

from __future__ import annotations

from ipu_isa.codec.disasm import slot_operations
from ipu_isa.codec.word import DecodedWord
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot

#: One label per slot instance, in layout order; ``*`` marks debug-only slots.
SLOT_LABELS = tuple(
    label + ("*" if SLOTS[slot].debug_only else "")
    for slot in LAYOUT_ORDER
    for label in SLOTS[slot].instance_labels()
)


def stage_operations(instruction: dict[str, int] | None) -> list[tuple[str, str]]:
    """Decode every slot, including idle NOPs, without executing any operation.

    Each operation is its mnemonic and bound operands in assembly operand order,
    unused fields omitted.
    """
    if instruction is None:
        return [(label, "NOP") for label in SLOT_LABELS]
    operations = slot_operations(DecodedWord(instruction).to_word())
    return [(label, text) for label, (_, text) in zip(SLOT_LABELS, operations, strict=True)]


def display_operation(operation: str) -> str:
    """Assembly-style casing and operand separators for the stage view."""
    mnemonic, _, operands = operation.partition(" ")
    if mnemonic.upper() == "NOP":
        return "IDLE / NOP"
    return mnemonic.upper() + (" " + ", ".join(operands.upper().split()) if operands else "")


# Physical stages group execution slots; LOAD is the XMEM sideband.
PIPELINE_STAGES = (
    ("CTRL", (*SLOTS[Slot.LR].instance_labels(), "COND", "BREAK")),
    ("MULT", ("MULT",)),
    ("ACC", ("ACC",)),
    ("AAQ", ("AAQ",)),
    ("STORE", ("STORE",)),
    ("XMEM input", ("LOAD",)),
    ("Simulation", ("ACC_STORE*",)),
)


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
