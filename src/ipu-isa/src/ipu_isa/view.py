"""The read-only, typed view of the ISA (E03-09).

Authors write nested dicts in ``instruction_spec.py`` (Q30). :func:`load_isa`
is the one function that reads them: it checks them against the schema
(E03-03 rules) and builds frozen records. Every consumer asks :func:`isa` for
an :class:`Isa` and reads records, not string-keyed dicts.

Later issues add fields to these records as they add keys to the schema.
The fault list and F-1's reserved encodings (E03-15) are in
``ipu_isa.faults``, derived from an :class:`Isa`. E03-14 added the register accesses
(``Operand.write``, ``Instruction.implicit_reads`` / ``implicit_writes``) and
checks them against the timing contract (``ipu_isa.timing``).
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ipu_isa.errors import IsaLookupError, IsaValueError
from ipu_isa.instruction_spec import (
    INSTRUCTION_SPEC,
    PSEUDO_INSTRUCTION_SPEC,
    InstructionDoc,
    syntax,
    validate_instruction_spec,
    validate_pseudo_instruction_spec,
)
from ipu_isa.operand_types import OPERAND_TYPES, OperandType
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot, SlotInfo
from ipu_isa.tags import Tag
from ipu_isa.timing import OPERAND_REGISTERS, RegisterAccess, View, validate_timing
from ipu_isa.union_layout import IsaLayout, compute_layout

#: When an operand's register is read (the authored ``"read"`` values): the
#: timing contract's :class:`ipu_isa.timing.View`.
Read = View


@dataclass(frozen=True, slots=True)
class Operand:
    """One operand of an instruction or pseudo-instruction."""

    name: str  # becomes the SV field <name>_<column>: wire format
    type: OperandType
    read: Read | None
    write: bool = False  # the register the instruction writes (E03-14)


@dataclass(frozen=True, slots=True)
class Instruction:
    """A real instruction: it has an opcode in its slot."""

    slot: Slot
    mnemonic: str
    opcode: int  # position in the slot; never written by hand
    operands: tuple[Operand, ...]
    tag: Tag
    doc: InstructionDoc
    execute_fn: str
    # Registers read or written that are not read/write operands (E03-14).
    implicit_reads: tuple[RegisterAccess, ...] = ()
    implicit_writes: tuple[RegisterAccess, ...] = ()

    def reads(self) -> tuple[RegisterAccess, ...]:
        """Every register this instruction reads, with its view: the ``read``
        operands (an ``LcrIdx`` names LR and CR), then :attr:`implicit_reads`."""
        operand_reads = tuple(
            RegisterAccess(register, operand.name, operand.read)
            for operand in self.operands
            if operand.read is not None
            for register in OPERAND_REGISTERS[operand.type.name]
        )
        return operand_reads + self.implicit_reads

    def writes(self) -> tuple[RegisterAccess, ...]:
        """Every register this instruction writes: the ``write`` operands, then
        :attr:`implicit_writes`."""
        operand_writes = tuple(
            RegisterAccess(register, operand.name)
            for operand in self.operands
            if operand.write
            for register in OPERAND_REGISTERS[operand.type.name]
        )
        return operand_writes + self.implicit_writes

    @property
    def syntax(self) -> str:
        """``MNEMONIC operand operand ...`` (E03-04)."""
        return syntax(self.mnemonic, [operand.name for operand in self.operands])

    def operand(self, name: str) -> Operand:
        """The operand called ``name``.

        Raises:
            IsaLookupError: the instruction has no such operand.
        """
        for operand in self.operands:
            if operand.name == name:
                return operand
        raise IsaLookupError(f"{self.slot}.{self.mnemonic} has no operand {name!r}")


def mnemonic_key(mnemonic: str) -> str:
    """The form two mnemonics are compared in: assembly accepts any case."""
    return mnemonic.upper()


def _mnemonic_index(instructions: tuple[Instruction, ...]) -> Mapping[str, Instruction]:
    return MappingProxyType({mnemonic_key(i.mnemonic): i for i in instructions})


@dataclass(frozen=True, slots=True)
class PseudoInstruction:
    """An assembler alias: expands to ``expands_to`` with ``args`` and has no opcode.

    Each of ``args`` is one of this pseudo's operand names or a literal
    register name (``CR0``).
    """

    mnemonic: str
    operands: tuple[Operand, ...]
    expands_to: Instruction
    args: tuple[str, ...]
    doc: InstructionDoc
    tag: Tag  # always ASSEMBLER


@dataclass(frozen=True, eq=False)
class Isa:
    """The whole ISA: slots, operand types, instructions and their layout."""

    slots: Mapping[Slot, SlotInfo]
    operand_types: Mapping[str, OperandType]
    layout: IsaLayout
    _instructions: Mapping[Slot, tuple[Instruction, ...]] = field(repr=False)
    _pseudos: tuple[PseudoInstruction, ...] = field(repr=False)
    #: Slot -> ``mnemonic_key`` -> instruction.
    _by_mnemonic: Mapping[Slot, Mapping[str, Instruction]] = field(repr=False)

    @property
    def spec_slot_order(self) -> tuple[Slot, ...]:
        """Slots in the order the instruction spec lists them.

        It fixes the order of generated enums (the assembler's opcode classes,
        the SV package); ``slots`` follows the slot registry instead.
        """
        return tuple(self._instructions)

    def instructions(self, slot: Slot) -> tuple[Instruction, ...]:
        """The instructions of ``slot``, in opcode order."""
        return self._instructions[slot]

    def instruction(self, slot: Slot, mnemonic: str) -> Instruction:
        """The instruction ``mnemonic`` of ``slot``, matched case-insensitively.

        Raises:
            IsaLookupError: ``slot`` has no such instruction.
        """
        found = self._by_mnemonic[slot].get(mnemonic_key(mnemonic))
        if found is not None:
            return found
        raise IsaLookupError(f"Instruction {mnemonic!r} not found in slot {str(slot)!r}")

    def by_opcode(self, slot: Slot, opcode: int) -> Instruction:
        """The instruction of ``slot`` with opcode ``opcode``.

        Raises:
            IsaLookupError: ``slot`` has no such opcode.
        """
        instructions = self._instructions[slot]
        if not 0 <= opcode < len(instructions):
            raise IsaLookupError(f"Slot {str(slot)!r} has no opcode {opcode}")
        return instructions[opcode]

    def has_instruction(self, mnemonic: str) -> bool:
        """True when some slot has an instruction ``mnemonic`` (case-insensitive)."""
        wanted = mnemonic_key(mnemonic)
        return any(wanted in by_mnemonic for by_mnemonic in self._by_mnemonic.values())

    def find_slot(self, mnemonic: str) -> Slot:
        """The one slot that has an instruction ``mnemonic`` (case-insensitive).

        Raises:
            IsaLookupError: no slot has it (a ``KeyError``).
            IsaValueError: several do, as ``NOP`` is in every slot (a ``ValueError``).
        """
        wanted = mnemonic_key(mnemonic)
        found = [slot for slot, by_mnemonic in self._by_mnemonic.items() if wanted in by_mnemonic]
        if not found:
            raise IsaLookupError(f"No slot has an instruction {mnemonic!r}")
        if len(found) > 1:
            raise IsaValueError(f"{mnemonic!r} is in several slots: {[str(s) for s in found]}")
        return found[0]

    def pseudos(self) -> tuple[PseudoInstruction, ...]:
        """Every pseudo-instruction, in authoring order."""
        return self._pseudos

    def pseudo(self, mnemonic: str, arity: int) -> PseudoInstruction | None:
        """The pseudo-instruction ``mnemonic`` taking ``arity`` operands, if any.

        Matched case-insensitively and by operand count, so a pseudo never
        shadows a real instruction of another arity with the same name.
        """
        wanted = mnemonic_key(mnemonic)
        for pseudo in self._pseudos:
            if mnemonic_key(pseudo.mnemonic) == wanted and len(pseudo.operands) == arity:
                return pseudo
        return None


def _operands(
    entries: list[dict[str, Any]], operand_types: Mapping[str, OperandType]
) -> tuple[Operand, ...]:
    return tuple(
        Operand(
            name=entry["name"],
            type=operand_types[entry["type"]],
            read=Read(entry["read"]) if "read" in entry else None,
            write=bool(entry.get("write", False)),
        )
        for entry in entries
    )


def _accesses(entries: list[dict[str, Any]]) -> tuple[RegisterAccess, ...]:
    return tuple(
        RegisterAccess(
            register=entry["register"],
            via_operand=entry.get("via_operand"),
            view=View(entry["view"]) if "view" in entry else None,
        )
        for entry in entries
    )


def load_isa(
    instruction_spec: Mapping[str, Mapping[str, Any]],
    pseudo_spec: Mapping[str, Mapping[str, Any]],
    slots: Mapping[Slot, SlotInfo],
    operand_types: Mapping[str, OperandType],
    layout_order: tuple[Slot, ...] = LAYOUT_ORDER,
) -> Isa:
    """Check the authoring dicts against the schema, then build the view.

    Pure: it reads only its arguments (the dicts and the slot and operand-type
    registries), so a test can load a modified copy.
    Raises :class:`ipu_isa.schema.SpecError` on a schema violation.
    """
    slot_metadata = {slot.value: {"description": info.description} for slot, info in slots.items()}
    validate_instruction_spec(
        dict(instruction_spec), slot_metadata, slots=slots, operand_types=operand_types
    )
    validate_pseudo_instruction_spec(
        dict(pseudo_spec), dict(instruction_spec), slots=slots, operand_types=operand_types
    )
    validate_timing(instruction_spec, slots)
    layout = compute_layout(instruction_spec, operand_types, slots, layout_order)

    instructions: dict[Slot, tuple[Instruction, ...]] = {}
    for slot_name, entries in instruction_spec.items():
        slot = Slot(slot_name)
        instructions[slot] = tuple(
            Instruction(
                slot=slot,
                mnemonic=mnemonic,
                opcode=opcode,
                operands=_operands(entry["operands"], operand_types),
                tag=entry.get("tag", slots[slot].tag),
                doc=entry["doc"],
                execute_fn=entry["execute_fn"],
                implicit_reads=_accesses(entry.get("implicit_reads", [])),
                implicit_writes=_accesses(entry.get("implicit_writes", [])),
            )
            for opcode, (mnemonic, entry) in enumerate(entries.items())
        )

    by_mnemonic = {slot: _mnemonic_index(entries) for slot, entries in instructions.items()}

    def target(expansion: Mapping[str, Any]) -> Instruction:
        return by_mnemonic[Slot(expansion["slot"])][mnemonic_key(str(expansion["instruction"]))]

    pseudos = tuple(
        PseudoInstruction(
            mnemonic=mnemonic,
            operands=_operands(entry["operands"], operand_types),
            expands_to=target(entry["expands_to"]),
            args=tuple(entry["expands_to"]["args"]),
            doc=entry["doc"],
            tag=entry.get("tag", Tag.ASSEMBLER),
        )
        for mnemonic, entry in pseudo_spec.items()
    )
    return Isa(
        slots=MappingProxyType(dict(slots)),
        operand_types=MappingProxyType(dict(operand_types)),
        layout=layout,
        _instructions=MappingProxyType(instructions),
        _pseudos=pseudos,
        _by_mnemonic=MappingProxyType(by_mnemonic),
    )


def authoring_copies() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Deep copies of ``(instruction spec, pseudo spec)``, to edit and pass to
    :func:`load_isa` (tests and E03-24's one-edit check). The originals stay
    private to ``ipu_isa``."""
    return copy.deepcopy(INSTRUCTION_SPEC), copy.deepcopy(PSEUDO_INSTRUCTION_SPEC)


@functools.cache
def isa() -> Isa:
    """The repository's ISA.

    Cached: the authoring dicts are never modified in place (a test that
    needs a different ISA copies them and calls :func:`load_isa`).
    """
    return load_isa(INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)
