"""The read-only, typed view of the ISA (E03-09).

Authors write nested dicts in ``instruction_spec.py`` (Q30). :func:`load_isa`
is the one function that reads them: it checks them against the schema
(E03-03 rules) and builds frozen records. Every consumer asks :func:`isa` for
an :class:`Isa` and reads records, not string-keyed dicts.

Later issues add fields to these records as they add keys to the schema:
formats and faults (E03-13, E03-15). E03-14 added the register accesses
(``Operand.write``, ``Instruction.implicit_reads`` / ``implicit_writes``) and
checks them against the timing contract (``ipu_isa.timing``).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

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
        for operand in self.operands:
            if operand.name == name:
                return operand
        raise KeyError(f"{self.slot}.{self.mnemonic} has no operand {name!r}")


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

    def instructions(self, slot: Slot) -> tuple[Instruction, ...]:
        """The instructions of ``slot``, in opcode order."""
        return self._instructions[slot]

    def instruction(self, slot: Slot, mnemonic: str) -> Instruction:
        """The instruction ``mnemonic`` of ``slot``, matched case-insensitively."""
        wanted = mnemonic.upper()
        for instruction in self._instructions[slot]:
            if instruction.mnemonic.upper() == wanted:
                return instruction
        raise KeyError(f"Instruction {mnemonic!r} not found in slot {str(slot)!r}")

    def by_opcode(self, slot: Slot, opcode: int) -> Instruction:
        """The instruction of ``slot`` with opcode ``opcode``."""
        instructions = self._instructions[slot]
        if not 0 <= opcode < len(instructions):
            raise KeyError(f"Slot {str(slot)!r} has no opcode {opcode}")
        return instructions[opcode]

    def find_slot(self, mnemonic: str) -> Slot:
        """The one slot that has an instruction ``mnemonic`` (case-insensitive).

        Raises ``KeyError`` if no slot has it and ``ValueError`` if several do
        (``NOP`` is in every slot).
        """
        wanted = mnemonic.upper()
        found = [
            slot
            for slot, instructions in self._instructions.items()
            if any(instruction.mnemonic.upper() == wanted for instruction in instructions)
        ]
        if not found:
            raise KeyError(f"No slot has an instruction {mnemonic!r}")
        if len(found) > 1:
            raise ValueError(f"{mnemonic!r} is in several slots: {[str(s) for s in found]}")
        return found[0]

    def pseudos(self) -> tuple[PseudoInstruction, ...]:
        """Every pseudo-instruction, in authoring order."""
        return self._pseudos

    def pseudo(self, mnemonic: str, arity: int) -> PseudoInstruction | None:
        """The pseudo-instruction ``mnemonic`` taking ``arity`` operands, if any.

        Matched case-insensitively and by operand count, so a pseudo never
        shadows a real instruction of another arity with the same name.
        """
        wanted = mnemonic.upper()
        for pseudo in self._pseudos:
            if pseudo.mnemonic.upper() == wanted and len(pseudo.operands) == arity:
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

    def target(expansion: Mapping[str, Any]) -> Instruction:
        slot = Slot(expansion["slot"])
        wanted = str(expansion["instruction"]).upper()
        return next(i for i in instructions[slot] if i.mnemonic.upper() == wanted)

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
    )


@functools.cache
def isa() -> Isa:
    """The repository's ISA.

    Cached: the authoring dicts are never modified in place (a test that
    needs a different ISA copies them and calls :func:`load_isa`).
    """
    return load_isa(INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)


# ---------------------------------------------------------------------------
# Lookup helpers with the return shapes of the old dict helpers, over isa(),
# so callers migrate one at a time (L34). E03-11 deletes them.
# ---------------------------------------------------------------------------


def _operand_dicts(operands: tuple[Operand, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": op.name,
            "type": op.type.name,
            **({"read": op.read.value} if op.read else {}),
            **({"write": True} if op.write else {}),
        }
        for op in operands
    ]


def _instruction_dict(instruction: Instruction) -> dict[str, Any]:
    return {
        "operands": _operand_dicts(instruction.operands),
        "doc": instruction.doc,
        "execute_fn": instruction.execute_fn,
    }


def get_opcode_for_instruction(slot_type: str, instruction_name: str) -> int:
    """Opcode of ``instruction_name`` in ``slot_type`` (case-insensitive)."""
    return isa().instruction(Slot(slot_type), instruction_name).opcode


def get_instruction(slot_type: str, instruction_name: str) -> dict[str, Any]:
    """``{"operands": [...], "doc": ..., "execute_fn": ...}`` of an instruction."""
    return _instruction_dict(isa().instruction(Slot(slot_type), instruction_name))


def get_instruction_by_opcode(slot_type: str, opcode_index: int) -> tuple[str, dict[str, Any]]:
    """``(mnemonic, instruction dict)`` of the instruction with ``opcode_index``."""
    instruction = isa().by_opcode(Slot(slot_type), opcode_index)
    return instruction.mnemonic, _instruction_dict(instruction)


def get_operand_names_and_types(slot_type: str, instruction_name: str) -> list[tuple[str, str]]:
    """``[(operand name, operand type name), ...]`` of an instruction."""
    instruction = isa().instruction(Slot(slot_type), instruction_name)
    return [(op.name, op.type.name) for op in instruction.operands]


def find_pseudo_instruction(name: str, arg_count: int) -> dict[str, Any] | None:
    """The pseudo-instruction dict for ``name`` with ``arg_count`` operands, or None."""
    pseudo = isa().pseudo(name, arg_count)
    if pseudo is None:
        return None
    return {
        "operands": _operand_dicts(pseudo.operands),
        "expands_to": {
            "slot": pseudo.expands_to.slot.value,
            "instruction": pseudo.expands_to.mnemonic,
            "args": list(pseudo.args),
        },
        "doc": pseudo.doc,
    }
