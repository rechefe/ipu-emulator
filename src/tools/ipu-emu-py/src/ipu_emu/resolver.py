"""The operand resolver seam: how an op's read operands become register values.

Issuing a slot op (:mod:`ipu_emu.issue`) replaces each ``read`` operand's
encoded index with the value of the register it names. That step has one
interface, :class:`OperandResolver`, and two adapters:

* :class:`FaithfulResolver` reads one operand at a time through the register
  file's public accessors (``get_lr``, ``get_cr``, ``get_register_bytes``).
* :class:`PackedResolver` (prototyping mode) reads LR and CR words with
  ``struct.unpack_from`` over :meth:`RegFile.packed_bank` views of the whole
  banks, taken once per register file, instead of one accessor call per read.

Both return equal operand dicts for every plan and encoding, errors included:
an index the packed path cannot serve goes to the public accessor, so it raises
what the faithful path raises. :mod:`ipu_emu.cycle` picks the adapter
(:func:`ipu_emu.cycle.resolver_for`); prototyping lives only at the adapter
(D7).

Which register file a read comes from is declared per operand (its read
timing): the cycle's snapshot for a cycle-start read, the live register file
otherwise.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ipu_emu.errors import EmulatorError
from ipu_isa.registers import get_mult_stage_map, get_register_sizes

if TYPE_CHECKING:
    from ipu_emu.ipu import SlotPlan
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.regfile import RegFile

# How a read operand resolves to a register value, by the banks its type names.
# ``ipu_emu.ipu`` assigns one of these to each read when it builds its plans.
READ_LR = 0
READ_CR = 1
READ_LCR = 2  # an LR below LR_REG_COUNT, else CR (index - LR_REG_COUNT)
READ_MULT_STAGE = 3  # R0 or R1 (MultStageReg)
READ_OTHER = 4  # passed through unchanged

_SIZES = get_register_sizes()
LR_REG_COUNT: int = _SIZES["lr"]["count"]
CR_REG_COUNT: int = _SIZES["cr"]["count"]

# MultStageReg encoding -> (register name, element index); 2 and 3 are reserved.
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()
# Wide-vector debug mode's R0/R1: the same two elements, 4 bytes per lane.
_WIDE_RA = "r_wide_debug"


def check_mult_stage(raw: int) -> None:
    """Raise unless *raw* encodes a mult-stage register (R0 = 0, R1 = 1).

    The one mult-stage operand check; both adapters call it.

    Raises:
        EmulatorError: *raw* is a reserved encoding.
    """
    if raw > 1:
        raise EmulatorError(f"Mult-stage operand must encode r0 (0) or r1 (1); got {raw}")


def resolve_mult_stage(raw: int, source: RegFile, wide: bool) -> bytearray:
    """A mult-stage operand's value: R0 or R1's bytes (``r_wide_debug``'s
    4-byte elements in wide-vector mode), so a handler decodes it in the
    active element format either way."""
    check_mult_stage(raw)
    reg_name, elem_idx = _MULT_STAGE_MAP[raw]
    return source.get_register_bytes(_WIDE_RA if wide else reg_name, elem_idx)


class OperandResolver(Protocol):
    """Resolves the read operands of one slot op."""

    def resolve(
        self, plan: SlotPlan, encoded: Mapping[str, int], snapshot: RegFile
    ) -> dict[str, Any]:
        """The handler's operands: *encoded* with each read resolved.

        Args:
            plan: The op's plan; ``plan.reads`` names the operands to resolve.
            encoded: The raw operand fields, by operand name. Not mutated.
            snapshot: The cycle's register snapshot, for cycle-start reads.
                Other reads come from the state's live register file.

        Returns:
            A new dict, by operand name.
        """
        ...


class FaithfulResolver:
    """Resolves reads one operand at a time through ``RegFile``'s public accessors.

    Args:
        state: The state whose live register file and wide-vector flag are read.
    """

    __slots__ = ("_state",)

    def __init__(self, state: IpuState) -> None:
        self._state = state

    def resolve(
        self, plan: SlotPlan, encoded: Mapping[str, int], snapshot: RegFile
    ) -> dict[str, Any]:
        """See :meth:`OperandResolver.resolve`."""
        state = self._state
        live = state.regfile
        operands = dict(encoded)
        for read in plan.reads:
            source = snapshot if read.from_snapshot else live
            raw = operands[read.name]
            code = read.code
            if code == READ_LR:
                operands[read.name] = source.get_lr(raw)
            elif code == READ_CR:
                operands[read.name] = source.get_cr(raw)
            elif code == READ_LCR:
                operands[read.name] = (
                    source.get_lr(raw) if raw < LR_REG_COUNT else source.get_cr(raw - LR_REG_COUNT)
                )
            elif code == READ_MULT_STAGE:
                operands[read.name] = resolve_mult_stage(
                    raw, source, state.codec.element_bytes != 1
                )
        return operands


_WORD = struct.Struct("<I").unpack_from
_BANK_COUNT = {READ_LR: LR_REG_COUNT, READ_CR: CR_REG_COUNT}


class PackedResolver:
    """Resolves LR and CR reads with ``struct.unpack_from`` over packed bank views.

    The :meth:`RegFile.packed_bank` views of each register file it reads (the
    live file and the cycle's reused snapshot buffer) are taken once and kept.
    An index outside the bank goes to the public accessor, so it raises exactly
    what :class:`FaithfulResolver` raises.

    Args:
        state: The state whose live register file and wide-vector flag are read.
    """

    __slots__ = ("_banks", "_state")

    #: Register files whose views are kept: the live file and a snapshot, with
    #: room for a replaced one. More than this and the cache starts over.
    _MAX_CACHED = 4

    def __init__(self, state: IpuState) -> None:
        self._state = state
        # id(register file) -> (the file, {read code: its bank's view}). The
        # file is held so its id cannot be reused while the entry exists.
        self._banks: dict[int, tuple[RegFile, dict[int, memoryview]]] = {}

    def _views(self, source: RegFile) -> dict[int, memoryview]:
        entry = self._banks.get(id(source))
        if entry is None or entry[0] is not source:
            if len(self._banks) >= self._MAX_CACHED:
                self._banks.clear()
            entry = (
                source,
                {READ_LR: source.packed_bank("lr"), READ_CR: source.packed_bank("cr")},
            )
            self._banks[id(source)] = entry
        return entry[1]

    def resolve(
        self, plan: SlotPlan, encoded: Mapping[str, int], snapshot: RegFile
    ) -> dict[str, Any]:
        """See :meth:`OperandResolver.resolve`."""
        state = self._state
        live = state.regfile
        operands = dict(encoded)
        for read in plan.reads:
            source = snapshot if read.from_snapshot else live
            raw = operands[read.name]
            code = read.code
            if code == READ_LCR:
                if raw < LR_REG_COUNT:
                    code = READ_LR
                else:
                    code, raw = READ_CR, raw - LR_REG_COUNT
            if code == READ_LR or code == READ_CR:
                if 0 <= raw < _BANK_COUNT[code]:
                    operands[read.name] = _WORD(self._views(source)[code], raw * 4)[0]
                else:  # the accessor raises its own error
                    operands[read.name] = (
                        source.get_lr(raw) if code == READ_LR else source.get_cr(raw)
                    )
            elif code == READ_MULT_STAGE:
                operands[read.name] = resolve_mult_stage(
                    raw, source, state.codec.element_bytes != 1
                )
        return operands
