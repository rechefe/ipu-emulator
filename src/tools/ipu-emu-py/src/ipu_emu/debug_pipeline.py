"""Read-only VLIW stage descriptions derived from assembler metadata."""
from __future__ import annotations

from ipu_as.compound_inst import CompoundInst
from ipu_common.instruction_spec import COMPOUND_LAYOUT_SLOT_ORDER, SLOT_COUNT, SLOT_UNIONS, is_hardware_slot

SLOT_LABELS = tuple(
    (slot.upper() + (str(index) if SLOT_COUNT[slot] > 1 else "")
     + ("*" if not is_hardware_slot(slot) else ""))
    for slot in COMPOUND_LAYOUT_SLOT_ORDER
    for index in range(SLOT_COUNT[slot])
)


def stage_operations(instruction: dict[str, int] | None) -> list[tuple[str, str]]:
    """Decode every slot, including idle NOPs, without executing any operation."""
    if instruction is None:
        return [(label, "NOP") for label in SLOT_LABELS]
    word = 0
    shift = 0
    for name, width in CompoundInst.get_fields():
        word |= (instruction.get(name, 0) & ((1 << width) - 1)) << shift
        shift += width
    operations = [part.strip() for part in CompoundInst.decode(word).split(";") if part.strip()]
    return list(zip(SLOT_LABELS, semantic_operations(operations), strict=True))


def display_operation(operation: str) -> str:
    """Assembly-style casing and operand separators for the stage view."""
    mnemonic, _, operands = operation.partition(" ")
    if mnemonic.upper() == "NOP":
        return "IDLE / NOP"
    return mnemonic.upper() + (" " + ", ".join(operands.upper().split()) if operands else "")


def semantic_operations(operations: list[str]) -> list[str]:
    """Select bound fields in assembly operand order, omitting unused bits."""
    slot_types = [slot for slot in COMPOUND_LAYOUT_SLOT_ORDER for _ in range(SLOT_COUNT[slot])]
    if len(operations) != len(slot_types):
        return operations
    result = []
    for slot, operation in zip(slot_types, operations, strict=True):
        tokens = operation.split()
        if not tokens:
            result.append(operation)
            continue
        mnemonic = tokens[0]
        bindings = SLOT_UNIONS[slot].opcode_bindings.get(mnemonic)
        if bindings is None:
            result.append(operation)
            continue
        operands = [tokens[index + 1] for index, _ in bindings]
        result.append(" ".join([mnemonic, *operands]))
    return result


# Physical stages group execution slots; LOAD is the XMEM sideband.
PIPELINE_STAGES = (
    ("CTRL", ("LR0", "LR1", "LR2", "COND", "BREAK")),
    ("MULT", ("MULT",)),
    ("ACC", ("ACC",)),
    ("AAQ", ("AAQ",)),
    ("STORE", ("STORE",)),
    ("XMEM input", ("LOAD",)),
    ("Simulation", ("ACC_STORE*",)),
)


def pipeline_occupancy(instruction: dict[str, int] | None) -> list[tuple[str, list[tuple[str, str]]]]:
    """Commands ready in each physical stage at the paused cycle boundary.

    NOPs are empty stages, not previously completed commands held in flight.
    Several control operations may occupy CTRL concurrently.
    """
    slots = dict(stage_operations(instruction))
    return [(stage, [(slot, slots[slot]) for slot in members
                     if slots[slot].split()[0].upper() != "NOP"])
            for stage, members in PIPELINE_STAGES]
