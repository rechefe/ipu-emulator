"""The hardware fault list F-1 to F-9 and the arithmetic events, as data (E03-15).

This mirrors the proposed fault list in ``docs/content/requirements/faults.md``
(status Proposed, OQ-1) and does not change it: decisions Q52, Q53 and Q59.
The emulator's fault model (E04-19), the commit-record schema (E04-29), the
docs and a future RTL testbench read the IDs from here, so they all use the
same numbers.

* A fault's number is its priority: when one word raises several faults, the
  lowest number is recorded (IPU-FLT-8).
* Every fault has an enable bit, set after reset (IPU-FLT-5, IPU-FLT-6).
* F-8 (word stall limit) is not modelled: the emulator has no stall model (Q59).
* Accumulator overflow, NaN results and quantization saturation are
  :class:`ArithmeticEvent` values, not faults: they never halt and each has a
  sticky flag (Q53).

F-1's reserved encodings are not listed by hand: :func:`reserved_encodings`
derives them from the ISA (the opcode count and width of each HARDWARE slot
and the value domains of its operand types).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from ipu_isa.operand_types import OperandKind, OperandType
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag

if TYPE_CHECKING:
    from ipu_isa.view import Instruction, Isa

__all__ = [
    "FAULTS",
    "RESERVED_MEMBER_PREFIX",
    "ArithmeticEvent",
    "Detector",
    "FaultId",
    "FaultInfo",
    "ReservedEncodings",
    "ReservedOperand",
    "instruction_faults",
    "reserved_encodings",
    "reserved_operand_codes",
]


class FaultId(IntEnum):
    """A hardware fault. The value is the fault number and its priority:
    the lowest number wins when one word raises several (IPU-FLT-8)."""

    F1 = 1
    F2 = 2
    F3 = 3
    F4 = 4
    F5 = 5
    F6 = 6
    F7 = 7
    F8 = 8
    F9 = 9

    @property
    def label(self) -> str:
        """``"F-3"``: the ID as the requirements write it."""
        return f"F-{self.value}"


class Detector(str, Enum):
    """The unit that detects a fault (IPU-FLT-2).

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Slot``.
    """

    CTRL = "ctrl"
    CACHE_AGENT = "cache_agent"

    def __str__(self) -> str:
        return self.value


class ArithmeticEvent(str, Enum):
    """An arithmetic event in ACC or AAQ: never halts, sets a sticky flag (Q53).

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Slot``.
    """

    ACC_OVERFLOW = "acc_overflow"
    NAN_RESULT = "nan_result"
    QUANT_SATURATION = "quant_saturation"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FaultInfo:
    """One fault of the fault list (faults.md): what it is and where it is caught."""

    id: FaultId
    summary: str  # the "Illegal state" cell of faults.md, verbatim
    detected_in: Detector
    modelled: bool  # False only for F8: the emulator has no stall model (Q59)
    enabled_at_reset: bool  # True for every fault (IPU-FLT-6)


def _fault(
    fault_id: FaultId, summary: str, detected_in: Detector, *, modelled: bool = True
) -> FaultInfo:
    return FaultInfo(fault_id, summary, detected_in, modelled, enabled_at_reset=True)


#: Every fault, in number (priority) order.
FAULTS: Mapping[FaultId, FaultInfo] = MappingProxyType(
    {
        info.id: info
        for info in (
            _fault(
                FaultId.F1,
                "A slot holds a reserved opcode or a reserved operand encoding.",
                Detector.CTRL,
            ),
            _fault(
                FaultId.F2,
                "CTRL fetches at a PC at or beyond the program length, by sequential flow, "
                "branch or `BR`.",
                Detector.CTRL,
            ),
            _fault(FaultId.F3, "Two LR lanes of one word write the same LR.", Detector.CTRL),
            _fault(
                FaultId.F4,
                "The index of `LD.MULT.CYCLIC` is not 0, 128, 256 or 384.",
                Detector.CTRL,
            ),
            _fault(
                FaultId.F5,
                "`ACC.RESHAPE` has a mask above 8, or a participating source or destination "
                "index of 128 or more.",
                Detector.CTRL,
            ),
            _fault(
                FaultId.F6,
                "A load or a store addresses a table that is not configured, or a row beyond "
                "the table size.",
                Detector.CACHE_AGENT,
            ),
            _fault(
                FaultId.F7,
                "A load addresses a write-only table, or a store addresses a read-only table.",
                Detector.CACHE_AGENT,
            ),
            _fault(
                FaultId.F8,
                "A word stalls for longer than a programmable limit.",
                Detector.CACHE_AGENT,
                modelled=False,
            ),
            _fault(
                FaultId.F9,
                "A kernel runs for more cycles than a programmable limit.",
                Detector.CTRL,
            ),
        )
    }
)


# ---------------------------------------------------------------------------
# F-1: reserved encodings, derived from the ISA
# ---------------------------------------------------------------------------

#: An enum member whose name starts with this pads the field and has no
#: meaning (``HorizontalStride.reserved3``): its code is reserved.
RESERVED_MEMBER_PREFIX = "reserved"


@dataclass(frozen=True, slots=True)
class ReservedOperand:
    """The reserved codes of one operand type in one slot."""

    operand_type: str  # "MultStageReg"
    bits: int  # field width
    codes: tuple[int, ...]  # encoded field values with no meaning, ascending
    operands: tuple[str, ...]  # operand names of this type in the slot, first-use order


@dataclass(frozen=True, slots=True)
class ReservedEncodings:
    """F-1 for one HARDWARE slot: its reserved opcodes and operand codes."""

    slot: Slot
    opcode_bits: int
    opcodes: tuple[int, ...]  # opcode values no instruction has, ascending
    operands: tuple[ReservedOperand, ...]  # in operand-type first-use order

    @property
    def empty(self) -> bool:
        """True when every encoding of the slot has a meaning."""
        return not self.opcodes and not self.operands


def reserved_operand_codes(operand_type: OperandType, bits: int) -> tuple[int, ...]:
    """The codes of a ``bits``-wide field of ``operand_type`` that mean nothing.

    * REGISTER and ENUM: codes past the last member, and members named
      ``reserved...``.
    * IMMEDIATE and LABEL with a value range narrower than the field: the codes
      past the range's size. The range is encoded from code 0 upward
      (``INC.MODP2`` stores ``k - 1``), so codes ``size`` and above are
      reserved. A range as wide as the field, or wider (both signed and
      unsigned spellings accepted), reserves nothing.
    * An immediate without a range (its width set by union packing) reserves
      nothing here; F-5 covers the ``ACC.RESHAPE`` mask.
    """
    capacity = 1 << bits
    domain = operand_type.domain
    if operand_type.kind in (OperandKind.REGISTER, OperandKind.ENUM):
        named = [
            code
            for code, member in enumerate(domain.members)
            if member.startswith(RESERVED_MEMBER_PREFIX)
        ]
        return tuple(named) + tuple(range(len(domain.members), capacity))
    if domain.value_range is None:
        return ()
    low, high = domain.value_range
    size = high - low + 1
    return tuple(range(size, capacity))


def _slot_reserved(isa: Isa, slot: Slot) -> ReservedEncodings:
    instructions = isa.instructions(slot)
    opcode_bits = isa.layout.unions[slot].opcode_bits
    opcodes = tuple(range(len(instructions), 1 << opcode_bits))

    operand_names: dict[str, list[str]] = {}
    for instruction in instructions:
        for operand in instruction.operands:
            names = operand_names.setdefault(operand.type.name, [])
            if operand.name not in names:
                names.append(operand.name)

    operands = []
    for type_name, names in operand_names.items():
        bits = isa.layout.operand_bits[type_name]
        codes = reserved_operand_codes(isa.operand_types[type_name], bits)
        if codes:
            operands.append(ReservedOperand(type_name, bits, codes, tuple(names)))
    return ReservedEncodings(slot, opcode_bits, opcodes, tuple(operands))


def reserved_encodings(isa: Isa) -> Mapping[Slot, ReservedEncodings]:
    """F-1's reserved encodings of every HARDWARE slot, in layout order.

    Slots whose every encoding has a meaning are included with empty tuples.
    EMULATOR slots are not in the hardware word, so they have no entry.
    """
    return MappingProxyType(
        {slot: _slot_reserved(isa, slot) for slot in isa.layout.slots_with_tag(Tag.HARDWARE)}
    )


# ---------------------------------------------------------------------------
# Which faults an instruction can raise (for the instruction reference, E09-14)
# ---------------------------------------------------------------------------

#: The instructions F-4 and F-5 name in their own summaries.
_F4_MNEMONIC = "LD.MULT.CYCLIC"
_F5_MNEMONIC = "ACC.RESHAPE"


def instruction_faults(isa: Isa, instruction: Instruction) -> tuple[FaultId, ...]:
    """The faults one instruction can raise, in number order.

    Read from each fault's summary, not listed per instruction:

    * F-1: an operand whose type has reserved codes in its field.
    * F-2: a COND branch, which fetches its target (an instruction of the
      COND slot that takes an operand; ``HALT`` takes none and fetches nothing).
      F-2 by sequential flow belongs to a word, not an instruction.
    * F-3: an LR-slot instruction that writes an LR: two lanes of one word may
      write the same one.
    * F-4: ``LD.MULT.CYCLIC``; F-5: ``ACC.RESHAPE``.
    * F-6 and F-7: every LOAD and STORE instruction (they address a table).
    * F-8 and F-9 belong to a word or a kernel, never to one instruction.

    ``NOP`` and EMULATOR or ASSEMBLER instructions raise none: faults are
    hardware.

    Args:
        isa: The ISA the instruction belongs to.
        instruction: The instruction.

    Returns:
        Its fault IDs, ascending; empty for none.
    """
    if instruction.tag is not Tag.HARDWARE or instruction.mnemonic == "NOP":
        return ()
    slot = instruction.slot
    faults: set[FaultId] = set()
    for operand in instruction.operands:
        bits = isa.layout.operand_bits[operand.type.name]
        if reserved_operand_codes(operand.type, bits):
            faults.add(FaultId.F1)
    if slot is Slot.COND and instruction.operands:
        faults.add(FaultId.F2)
    if slot is Slot.LR and any(operand.write for operand in instruction.operands):
        faults.add(FaultId.F3)
    if instruction.mnemonic == _F4_MNEMONIC:
        faults.add(FaultId.F4)
    if instruction.mnemonic == _F5_MNEMONIC:
        faults.add(FaultId.F5)
    if slot in (Slot.LOAD, Slot.STORE):
        faults.update((FaultId.F6, FaultId.F7))
    return tuple(sorted(faults))
