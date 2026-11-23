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
from typing import TYPE_CHECKING, Any, Final, Protocol

from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import WIDE_REGISTERS
from ipu_isa.errors import IsaValueError
from ipu_isa.cr_halves import LcrHalf, lcr_half_code, lcr_half_value
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
READ_LCR_HALF = 5  # an LR, a CR or a sign-extended CR half (LcrHalfIdx, ADR 0007)

_SIZES = get_register_sizes()
LR_REG_COUNT: int = _SIZES["lr"]["count"]
CR_REG_COUNT: int = _SIZES["cr"]["count"]

# MultStageReg encoding -> (register name, element index); 2 and 3 are reserved.
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()
# A 4-byte format's R0/R1: the same two elements, 4 bytes per lane.
_WIDE_RA = WIDE_REGISTERS["r"]


def check_mult_stage(raw: int) -> None:
    """Raise unless *raw* encodes a mult-stage register (R0 = 0, R1 = 1).

    The one mult-stage operand check; both adapters call it. An encoded word
    never reaches it with a reserved code: ``cycle.schedule_word`` makes that
    word fault F-1 at fetch (E04-19). It guards words built by hand.

    Raises:
        EmulatorError: *raw* is a reserved encoding.
    """
    if raw > 1:
        raise EmulatorError(f"Mult-stage operand must encode r0 (0) or r1 (1); got {raw}")


def resolve_lcr_half(raw: int, source: RegFile) -> int:
    """An ``LcrHalfIdx`` operand's value: the LR or CR word, or a CR half
    sign-extended to 32 bits (returned unsigned), as the LR and COND
    instructions read it (ADR 0007)."""
    operand = lcr_half_code(raw)
    word = source.get_lr(operand.index) if operand.bank == "lr" else source.get_cr(operand.index)
    return lcr_half_value(operand, word)


def resolve_mult_stage(raw: int, source: RegFile, wide: bool) -> bytearray:
    """A mult-stage operand's value: R0 or R1's bytes (``r_wide_debug``'s
    4-byte elements in FP32), so a handler decodes it in the active element
    format either way."""
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
        operands: dict[str, Any] = dict(encoded)
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
            elif code == READ_LCR_HALF:
                operands[read.name] = resolve_lcr_half(raw, source)
            elif code == READ_MULT_STAGE:
                operands[read.name] = resolve_mult_stage(
                    raw, source, state.codec.element_bytes != 1
                )
        return operands


_WORD: Final = struct.Struct("<I").unpack_from
_BANK_COUNT: Final = {READ_LR: LR_REG_COUNT, READ_CR: CR_REG_COUNT}


def _lcr_half_table() -> dict[int, LcrHalf]:
    """Every valid ``LcrHalfIdx`` code's register (``lcr_half_code`` is a pure function of it)."""
    table: dict[int, LcrHalf] = {}
    code = 0
    while True:
        try:
            table[code] = lcr_half_code(code)
        except IsaValueError:
            return table
        code += 1


_LCR_HALVES: Final = _lcr_half_table()

# How one read of a decoded op resolves (PackedResolver's per-word plan): a
# packed LR/CR word, a sign-extended CR half of one, or a mult-stage register.
_STEP_WORD: Final = 0
_STEP_HALF: Final = 1
_STEP_MULT: Final = 2

#: (operand name, from the snapshot, step kind, bank read code, byte offset or
#: mult-stage code, the LcrHalf a half step sign-extends).
_Step = tuple[str, bool, int, int, int, "LcrHalf | None"]


def _read_steps(plan: SlotPlan, encoded: Mapping[str, int]) -> tuple[_Step, ...] | None:
    """The reads of (plan, encoded) as steps, or ``None`` when any of them needs
    the general path (an index the packed banks cannot serve, a code that is not
    an ``LcrHalfIdx`` register, a field that is not an int, a repeated or
    missing operand): it raises what it always raised."""
    steps: list[_Step] = []
    names: set[str] = set()
    for read in plan.reads:
        name = read.name
        if name in names or name not in encoded:
            return None
        names.add(name)
        raw_value: object = encoded[name]
        if type(raw_value) is not int:
            return None
        raw: int = encoded[name]
        code = read.code
        snap = bool(read.from_snapshot)
        if code == READ_LCR:
            if raw < LR_REG_COUNT:
                code = READ_LR
            else:
                code, raw = READ_CR, raw - LR_REG_COUNT
        if code == READ_LR or code == READ_CR:
            if not 0 <= raw < _BANK_COUNT[code]:
                return None
            steps.append((name, snap, _STEP_WORD, code, raw * 4, None))
        elif code == READ_LCR_HALF:
            operand = _LCR_HALVES.get(raw)
            if operand is None:
                return None
            bank = READ_LR if operand.bank == "lr" else READ_CR
            if operand.bank not in ("lr", "cr") or not 0 <= operand.index < _BANK_COUNT[bank]:
                return None
            steps.append((name, snap, _STEP_HALF, bank, operand.index * 4, operand))
        elif code == READ_MULT_STAGE:
            steps.append((name, snap, _STEP_MULT, 0, raw, None))
        # any other read code passes the field through, as resolve() does
    return tuple(steps)


class PackedResolver:
    """Resolves LR and CR reads with ``struct.unpack_from`` over packed bank views.

    The :meth:`RegFile.packed_bank` views of each register file it reads (the
    live file and the cycle's reused snapshot buffer) are taken once and kept.
    An index outside the bank goes to the public accessor, so it raises exactly
    what :class:`FaithfulResolver` raises.

    Args:
        state: The state whose live register file and wide-vector flag are read.
    """

    __slots__ = ("_banks", "_live", "_live_views", "_snap", "_snap_views", "_state", "_steps")

    #: Register files whose views are kept: the live file and a snapshot, with
    #: room for a replaced one. More than this and the cache starts over.
    _MAX_CACHED = 4

    def __init__(self, state: IpuState) -> None:
        self._state = state
        # id(register file) -> (the file, {read code: its bank's view}). The
        # file is held so its id cannot be reused while the entry exists.
        self._banks: dict[int, tuple[RegFile, dict[int, memoryview]]] = {}
        # id(encoded operands) -> (them, the plan, their read steps or None).
        # The encoded dict is the word's schedule's own (never mutated); the
        # entry holds it, so its id is not reused while the entry exists.
        self._steps: dict[int, tuple[Mapping[str, int], SlotPlan, tuple[_Step, ...] | None]] = {}
        # The views of the last snapshot and the last live file resolve() read
        # (those of _views(), which the files are held by): the same two files
        # are read cycle after cycle.
        self._snap: RegFile | None = None
        self._snap_views: dict[int, memoryview] = {}
        self._live: RegFile | None = None
        self._live_views: dict[int, memoryview] = {}

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
        """See :meth:`OperandResolver.resolve`.

        Each op's reads are decoded once (:func:`_read_steps`, cached per
        encoded-operand dict); only the register reads run per call. An op the
        steps cannot serve takes :meth:`_resolve_general`, the same result.
        """
        entry = self._steps.get(id(encoded))
        if entry is None or entry[0] is not encoded or entry[1] is not plan:
            if len(self._steps) >= 1 << 16:
                self._steps.clear()
            entry = (encoded, plan, _read_steps(plan, encoded))
            self._steps[id(encoded)] = entry
        steps = entry[2]
        if steps is None:
            return self._resolve_general(plan, encoded, snapshot)
        operands: dict[str, Any] = dict(encoded)
        if not steps:
            return operands
        state = self._state
        live = state.regfile
        views_snap: dict[int, memoryview] | None = None
        views_live: dict[int, memoryview] | None = None
        for name, snap, kind, bank, offset, half in steps:
            if kind == _STEP_MULT:
                operands[name] = resolve_mult_stage(
                    offset, snapshot if snap else live, state.codec.element_bytes != 1
                )
                continue
            if snap:
                if views_snap is None:
                    if snapshot is self._snap:
                        views_snap = self._snap_views
                    else:
                        views_snap = self._views(snapshot)
                        self._snap, self._snap_views = snapshot, views_snap
                views = views_snap
            else:
                if views_live is None:
                    if live is self._live:
                        views_live = self._live_views
                    else:
                        views_live = self._views(live)
                        self._live, self._live_views = live, views_live
                views = views_live
            word: int = _WORD(views[bank], offset)[0]
            if kind == _STEP_WORD:
                operands[name] = word
            else:
                assert half is not None
                operands[name] = lcr_half_value(half, word)
        return operands

    def _resolve_general(
        self, plan: SlotPlan, encoded: Mapping[str, int], snapshot: RegFile
    ) -> dict[str, Any]:
        """:meth:`resolve` one read at a time (every case, errors included)."""
        state = self._state
        live = state.regfile
        operands: dict[str, Any] = dict(encoded)
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
            elif code == READ_LCR_HALF:
                operands[read.name] = resolve_lcr_half(raw, source)
            elif code == READ_MULT_STAGE:
                operands[read.name] = resolve_mult_stage(
                    raw, source, state.codec.element_bytes != 1
                )
        return operands
