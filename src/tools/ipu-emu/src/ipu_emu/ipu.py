"""The IPU execution engine: construction, plan building and handler binding.

This module holds:

- the per-(slot, opcode) dispatch plans (:class:`SlotPlan`), built once at
  import from the ISA view;
- the binding of every spec instruction to its handler, the function
  ``execute_<name>`` of its slot's module in :mod:`ipu_emu.slots` (E04-14);
- the :class:`Ipu` engine, which owns a :class:`~ipu_emu.cycle.Cycle` and an
  :class:`~ipu_emu.issue.Issuer` and runs no instruction itself.

Each handler receives NAMED operands (matching the instruction_spec operand
names), not raw instruction dicts: the issuer extracts operand values from the
instruction word and passes them as keyword arguments. Plan building checks
that the handler exists and that its keyword-only parameters are exactly the
spec's operand names.
"""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable, Mapping
from typing import NamedTuple

from ipu_emu.errors import ConfigurationError, EmulatorError
from ipu_emu.geometry import (
    NARROW_MAX_ROW,
    R_CYCLIC_SIZE,
    R_CYCLIC_VALID_INDICES,
    R_REG_SIZE,
    xmem_element_width_bytes,
    xmem_row_size_bytes,
)
from ipu_emu.ipu_state import IpuState
from ipu_emu.isa_alias_spec import SCALAR_MULT_OPERANDS, aliases_for
from ipu_emu.issue import Issuer
from ipu_emu.numerics import LaneCodec
from ipu_emu.regfile import RegFile
from ipu_emu.resolver import READ_CR, READ_LCR, READ_LR, READ_MULT_STAGE, READ_OTHER
from ipu_emu.slots import HANDLER_MODULES
from ipu_emu.slots.mult import SCALAR_CR, SCALAR_LCR, SCALAR_NONE
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.codec.word import field_name, opcode_field
from ipu_isa.faults import reserved_operand_codes
from ipu_isa.geometry import LANES
from ipu_isa.operand_types import OPERAND_TYPES, OperandKind, OperandType
from ipu_isa.registers import (
    REGISTER_DEFINITIONS,
    get_register_sizes,
)
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER
from ipu_isa.union_layout import SlotUnion
from ipu_isa.view import Instruction, Read, isa

# Re-exported: callers read row geometry and register sizes from here.
__all__ = [
    "LANES",
    "LR_REG_COUNT",
    "NARROW_MAX_ROW",
    "R_ACC_SIZE",
    "R_CYCLIC_SIZE",
    "R_CYCLIC_VALID_INDICES",
    "R_REG_SIZE",
    "XMEM_ADDRESSABLE_ROWS",
    "EmulatorError",
    "Ipu",
    "SlotPlan",
    "addressable_rows",
    "build_plan",
    "plans_for",
    "read_views",
    "xmem_element_width_bytes",
    "xmem_row_size_bytes",
]

# ---------------------------------------------------------------------------
# Constants — derived from the single source of truth in ipu-isa
# ---------------------------------------------------------------------------

_reg_sizes = get_register_sizes()

# Register dimensions — from REGISTER_DEFINITIONS via get_register_sizes()
LR_REG_COUNT = _reg_sizes["lr"]["count"]
R_ACC_SIZE = _reg_sizes["r_acc"]["size_bytes"]

# LANES (from ipu_isa.geometry) is the number of vector lanes: 128 in every
# element format. An element is 1 byte in an 8-bit format and 4 bytes in FP32
# and INT32, so LANES must never be confused with a byte count — use it for
# lane loop bounds, mask bit-widths, and lane-indexed lists/tuples in every
# format. R_REG_SIZE remains a byte count (the "r" register's size); it
# coincides with LANES only in an 8-bit format.

# XMEM row geometry (``xmem_element_width_bytes``, ``xmem_row_size_bytes``,
# ``NARROW_MAX_ROW``, ``xmem_row_addr``) lives in ``ipu_emu.geometry``, shared
# by the handlers and the observers; the first three are re-exported here.
# The FP32/INT32 formats reach the whole allocation in 512-byte rows.
XMEM_ADDRESSABLE_ROWS = XMEM_SIZE_BYTES // (LANES * 4)


def addressable_rows(state: IpuState) -> int:
    """Return how many XMEM rows ``.asm`` may address in the active mode."""
    return XMEM_ADDRESSABLE_ROWS if xmem_element_width_bytes(state) != 1 else NARROW_MAX_ROW


# ---------------------------------------------------------------------------
# Operand extraction: maps instruction_spec operand names → inst dict field keys
# ---------------------------------------------------------------------------

# The slots a word holds once, in registry order. LR's sub-slots are read per
# instance. Field names come from the ISA's word layout (field_name,
# opcode_field), never built here.
_SINGLE_INSTANCE_SLOTS = tuple(slot for slot, info in SLOTS.items() if info.count == 1)


def _build_field_map_for_instruction_union(
    slot: Slot,
    instance: int,
    slot_union: SlotUnion,
    inst_name: str,
) -> dict[str, str]:
    """Build operand_name → inst dict field_key mapping using union bindings."""
    return {
        operand_name: field_name(slot, instance, field_idx + 1)  # +1: field 0 is the opcode
        for field_idx, operand_name in slot_union.opcode_bindings.get(inst_name, [])
    }


def _build_all_instruction_field_maps() -> dict[tuple, dict[str, str]]:
    """Precompute operand → field_key mappings for ALL instructions.

    Returns a dict keyed by:
      (slot_type, inst_name)           — for single-instance slots
      (slot_type, inst_name, slot_idx) — for each instance of a multi-instance
                                         slot (LR: 0 .. count - 1)
    """
    result: dict[tuple, dict[str, str]] = {}
    unions = isa().layout.unions
    for slot, info in SLOTS.items():
        if info.count != 1:
            continue
        for instruction in isa().instructions(slot):
            result[(slot.value, instruction.mnemonic)] = _build_field_map_for_instruction_union(
                slot, 0, unions[slot], instruction.mnemonic
            )

    # Multi-instance slots (LR) read each instance from its own field prefix.
    for slot, info in SLOTS.items():
        if info.count == 1:
            continue
        for instruction in isa().instructions(slot):
            for slot_idx in range(info.count):
                result[(slot.value, instruction.mnemonic, slot_idx)] = (
                    _build_field_map_for_instruction_union(
                        slot, slot_idx, unions[slot], instruction.mnemonic
                    )
                )

    return result


# Precomputed at import time — no per-call overhead
_INSTRUCTION_FIELD_MAP = _build_all_instruction_field_maps()


def _build_read_operand_types() -> dict[tuple[str, str], dict[str, tuple[OperandType, Read]]]:
    """Precompute which operands need auto-resolution for each instruction.

    Returns dict keyed by (slot_type, inst_name) → {operand_name: (operand_type, read)}
    for the operands the ISA marks as read (``Read.CYCLE_START`` or ``Read.FORWARDED``).
    """
    result: dict[tuple[str, str], dict[str, tuple[OperandType, Read]]] = {}
    for slot in SLOTS:
        for instruction in isa().instructions(slot):
            read_ops = {op.name: (op.type, op.read) for op in instruction.operands if op.read}
            if read_ops:
                result[(slot.value, instruction.mnemonic)] = read_ops
    return result


_INSTRUCTION_READ_TYPES = _build_read_operand_types()


# ---------------------------------------------------------------------------
# Dispatch plans: everything about an instruction that does not depend on the
# instruction *word*, resolved once at import time.
#
# Dispatch runs nine times per emulated cycle, so anything it recomputes is
# multiplied by nine per cycle: building the opcode field name with an f-string,
# looking the instruction up by position in a freshly built list of keys,
# fetching the handler with getattr, and choosing a stats counter by string
# comparison. All of that is a function of (slot_type, opcode) alone.
# ---------------------------------------------------------------------------

# Register bank of each register name the assembler accepts ("lr3" -> "lr").
_BANK_OF_NAME = {
    name: bank
    for bank, definition in REGISTER_DEFINITIONS.items()
    for name in definition.get("assembler_values", ())
}


def _banks(operand_type: OperandType) -> frozenset[str | None]:
    """The register banks an operand type's domain names; None for a name that
    is not a register of one bank (an LRD pair)."""
    return frozenset(_BANK_OF_NAME.get(name) for name in operand_type.domain.members)


# How a read operand resolves, by the banks its type can name: a read code of
# ipu_emu.resolver, whose adapters do the resolving.
_READ_BY_BANKS = {
    frozenset({"lr"}): READ_LR,
    frozenset({"cr"}): READ_CR,
    frozenset({"lr", "cr"}): READ_LCR,
    frozenset({"r"}): READ_MULT_STAGE,
}

# Read code of every operand type, from its domain in the operand-type registry.
_READ_CODES = {
    name: _READ_BY_BANKS.get(_banks(operand_type), READ_OTHER)
    for name, operand_type in OPERAND_TYPES.items()
}


class _OperandRead(NamedTuple):
    """One operand that dispatch resolves to a register value."""

    name: str
    op_type: OperandType
    code: int
    from_snapshot: bool


_SCALAR_KINDS = {READ_LCR: SCALAR_LCR, READ_CR: SCALAR_CR}  # by read code


class _AliasCheck(NamedTuple):
    """One ISA_ALIAS_SPEC match, pre-resolved for the dispatch hot path.

    Every match either fires unconditionally (``hardware_only``) or requires the
    multiplicand to be 1.0 — the spec validator enforces exactly one of those —
    so the identity test itself is shared with the counters rather than stored
    per check.
    """

    alias_id: str
    imm_operand: str | None  # operand whose immediate must differ
    imm_ne: int
    co_field: str | None  # opcode field of the co-issued slot
    co_opcodes: frozenset  # ...and the opcodes that satisfy it
    hardware_only: bool  # matches unconditionally
    vector_only: bool
    masked_window: bool


class SlotPlan(NamedTuple):
    """Everything dispatch needs for one (slot, opcode), precomputed."""

    inst_name: str
    fn: Callable  # the handler: execute_<name>(ipu, **operands), from ipu_emu.slots
    fields: tuple[tuple[str, str], ...]  # (operand name, inst field key)
    reads: tuple[_OperandRead, ...]
    stat: str | None  # RunStats counter to bump
    write_operand: str | None  # LR slot: name of the target
    write_is_lrd: bool  # ...and whether it is an LRD pair
    is_mult: bool  # mult slot, and not its NOP
    scalar_operand: str | None  # mult slot: scalar multiplicand
    scalar_kind: int  # ...and how it is addressed
    alias_checks: tuple[_AliasCheck, ...]  # empty for almost every plan
    mask_cr_operand: str | None  # mult slot: the CR whose dstructure the lane mask uses
    # Every register the instruction reads, by (register, via operand), to
    # whether the read sees the cycle-start snapshot (else the live file).
    # From its ``read`` operands and ``implicit_reads``; a register whose
    # declared reads all share one view also has a ``(register, None)`` key.
    # ``CycleView`` serves handler reads from it (E04-12).
    views: Mapping[tuple[str, str | None], bool]
    # F-1: (operand name, field capacity, reserved codes) of every operand. The
    # union column can be wider than the operand type, so the capacity is the
    # type's width, not the column's.
    operand_checks: tuple[tuple[str, int, frozenset[int]], ...] = ()
    # The fault predicate ``check_<name>(ipu, **operands) -> (FaultId, detail)
    # | None`` beside the handler in its slot module, when the instruction can
    # fault on register values (E04-20); ``None`` for every other.
    check: Callable | None = None


def _operand_checks(instruction: Instruction) -> tuple[tuple[str, int, frozenset[int]], ...]:
    """Each operand's F-1 check: codes at or past its type's width, or reserved."""
    operand_bits = isa().layout.operand_bits
    checks = []
    for op in instruction.operands:
        bits = operand_bits[op.type.name]
        checks.append((op.name, 1 << bits, frozenset(reserved_operand_codes(op.type, bits))))
    return tuple(checks)


# Which RunStats counter a non-NOP instruction in each slot bumps.
_SLOT_STAT = {
    Slot.MULT: "mult_active_cycles",
    Slot.ACC: "acc_active_cycles",
    Slot.LOAD: "xmem_reads",
    Slot.STORE: "xmem_writes",
    Slot.ACC_STORE: "xmem_writes",
}


def _compile_alias_checks(slot_type: str, inst_name: str) -> tuple[_AliasCheck, ...]:
    """Lower this instruction's ISA_ALIAS_SPEC matches into runtime checks.

    Everything a constraint needs — operand names, literals, and the co-issued
    slot's opcode field and satisfying opcodes — is resolved here, once at
    import, so dispatch only compares integers and reads two dict keys.
    """
    checks = []
    for alias_id, match in aliases_for(slot_type, inst_name):
        imm_operand = imm_ne = None
        co_field, co_opcodes = None, frozenset()
        hardware_only = False
        vector_only = masked_window = False
        for constraint in match["where"]:
            if "scalar_is" in constraint:
                pass  # the identity test is shared; see _AliasCheck
            elif "immediate_ne" in constraint:
                imm_operand = constraint["operand"]
                imm_ne = constraint["immediate_ne"]
            elif "co_slot" in constraint:
                co_slot = constraint["co_slot"]
                co_field = opcode_field(Slot(co_slot))
                co_opcodes = frozenset(
                    isa().instruction(Slot(co_slot), name).opcode
                    for name in constraint["instruction_in"]
                )
            elif constraint.get("hardware_only"):
                hardware_only = True
            elif constraint.get("vector_ones"):
                vector_only = True
            elif constraint.get("masked_window"):
                masked_window = True
        checks.append(
            _AliasCheck(
                alias_id=alias_id,
                imm_operand=imm_operand,
                imm_ne=0 if imm_ne is None else imm_ne,
                co_field=co_field,
                co_opcodes=co_opcodes,
                hardware_only=hardware_only,
                vector_only=vector_only,
                masked_window=masked_window,
            )
        )
    return tuple(checks)


def build_plan(instruction: Instruction, field_map: dict[str, str]) -> SlotPlan:
    """The :class:`SlotPlan` of one instruction, whose fields sit at *field_map*'s keys.

    Raises:
        ConfigurationError: The spec names a handler its slot module does not have,
            one whose keyword-only parameters are not the operand names, or
            an LR write target that is not an LR or an LRD pair.
    """
    slot, inst_name = instruction.slot, instruction.mnemonic
    reads = tuple(
        _OperandRead(
            name=name,
            op_type=op_type,
            code=_READ_CODES[op_type.name],
            from_snapshot=(read is Read.CYCLE_START),
        )
        for name, (op_type, read) in _INSTRUCTION_READ_TYPES.get(
            (slot.value, inst_name), {}
        ).items()
    )
    # The LR slot needs the write target to detect two sub-instructions writing
    # the same register; the spec declares it with "write": True (E03-14).
    write_operand, write_is_lrd = None, False
    if slot is Slot.LR:
        for op in instruction.operands:
            if op.write:
                # An LR write target is an LR or an LRD pair; a pair's names
                # (lrd0, lrd2, ...) belong to no single register bank.
                banks = _banks(op.type)
                if op.type.kind is not OperandKind.REGISTER or banks not in (
                    frozenset({"lr"}),
                    frozenset({None}),
                ):
                    raise ConfigurationError(
                        f"{slot}/{inst_name}.{op.name}: not an LR or LRD target"
                    )
                write_operand, write_is_lrd = op.name, banks == frozenset({None})
                break
    fn = bind_handler(instruction)
    scalar_operand, scalar_kind = None, SCALAR_NONE
    if slot is Slot.MULT and inst_name in SCALAR_MULT_OPERANDS:
        scalar_operand = SCALAR_MULT_OPERANDS[inst_name]
        scalar_type = instruction.operand(scalar_operand).type
        scalar_kind = _SCALAR_KINDS[_READ_CODES[scalar_type.name]]
    return SlotPlan(
        inst_name=inst_name,
        fn=fn,
        fields=tuple(field_map.items()),
        reads=reads,
        stat=("mult_idle_cycles" if slot is Slot.MULT else None)
        if inst_name == "NOP"
        else _SLOT_STAT.get(slot),
        write_operand=write_operand,
        write_is_lrd=write_is_lrd,
        is_mult=(slot is Slot.MULT and inst_name != "NOP"),
        scalar_operand=scalar_operand,
        scalar_kind=scalar_kind,
        alias_checks=_compile_alias_checks(slot.value, inst_name),
        mask_cr_operand=_mask_cr_operand(instruction),
        views=read_views(instruction),
        operand_checks=_operand_checks(instruction),
        check=getattr(
            HANDLER_MODULES[slot], instruction.execute_fn.replace("execute_", "check_", 1), None
        ),
    )


def bind_handler(instruction: Instruction) -> Callable:
    """The handler *instruction*'s ``execute_fn`` names, from its slot's module.

    The handler is ``ipu_emu.slots.<slot>.execute_<name>(ipu, *, <operands>)``:
    one positional parameter (the engine) and a keyword-only parameter per
    operand, named exactly as in ``instruction_spec.py`` (CLAUDE.md, "Key
    Constraints").

    Args:
        instruction: The instruction, from the ISA view.

    Returns:
        The handler function.

    Raises:
        ConfigurationError: The slot module has no such function, or its parameters
            are not the engine followed by the operand names, keyword-only.
    """
    module = HANDLER_MODULES[instruction.slot]
    where = f"instruction_spec {instruction.slot}/{instruction.mnemonic} names handler"
    fn = getattr(module, instruction.execute_fn, None)
    if fn is None:
        raise ConfigurationError(
            f"{where} {module.__name__}.{instruction.execute_fn}, which does not exist "
            "(see CLAUDE.md, 'Adding an Instruction')"
        )
    params = list(inspect.signature(fn).parameters.values())
    keyword_only = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
    positional = [p for p in params if p.kind is not inspect.Parameter.KEYWORD_ONLY]
    operands = {op.name for op in instruction.operands}
    if len(positional) != 1 or keyword_only != operands:
        raise ConfigurationError(
            f"{where} {module.__name__}.{instruction.execute_fn}, whose parameters "
            f"{[p.name for p in params]} are not the engine and then, keyword-only, "
            f"the operands {sorted(operands)}"
        )
    return fn


def read_views(instruction: Instruction) -> Mapping[tuple[str, str | None], bool]:
    """Every read *instruction* declares, as :attr:`SlotPlan.views` holds it.

    Args:
        instruction: The instruction, from the ISA view.

    Returns:
        ``(register, via operand)`` to ``True`` for a cycle-start read and
        ``False`` for a forwarded one. A register whose declared reads all
        share one view is also keyed ``(register, None)``, so a handler that
        reads it without naming an operand gets that view.
    """
    views: dict[tuple[str, str | None], bool] = {}
    by_register: dict[str, set[bool]] = {}
    for access in instruction.reads():
        if access.view is None:
            continue
        from_snapshot = access.view is Read.CYCLE_START
        views[access.register, access.via_operand] = from_snapshot
        by_register.setdefault(access.register, set()).add(from_snapshot)
    for register, seen in by_register.items():
        if len(seen) == 1:
            views.setdefault((register, None), next(iter(seen)))
    return views  # a dict, not a read-only proxy: read on every handler read


def _mask_cr_operand(instruction: Instruction) -> str | None:
    """The operand naming the CR a MULT's lane mask partitions by, or ``None``.

    A MULT with a ``mask_offset`` operand masks its lanes: it reads the
    dstructure of ``dstructure_cr_idx`` when it has one, else of ``cr_idx``.
    """
    if instruction.slot is not Slot.MULT or instruction.mnemonic == "NOP":
        return None
    names = {op.name for op in instruction.operands}
    if "mask_offset" not in names:
        return None
    return "dstructure_cr_idx" if "dstructure_cr_idx" in names else "cr_idx"


def _build_slot_plans():
    """Build the opcode-indexed plan tables. Called once, at import.

    Returns ``(opcode_fields, plans, lr_plans, lr_opcode_fields)``.
    """
    opcode_fields = {slot_type: opcode_field(slot_type) for slot_type in _SINGLE_INSTANCE_SLOTS}
    plans = {
        slot_type: tuple(
            build_plan(instruction, _INSTRUCTION_FIELD_MAP[(slot_type, instruction.mnemonic)])
            for instruction in isa().instructions(Slot(slot_type))
        )
        for slot_type in _SINGLE_INSTANCE_SLOTS
    }
    # LR sub-slots differ only in which fields of the word they read.
    lr = SLOTS[Slot.LR]
    lr_plans = tuple(
        tuple(
            build_plan(
                instruction, _INSTRUCTION_FIELD_MAP[(lr.slot.value, instruction.mnemonic, slot_idx)]
            )
            for instruction in isa().instructions(Slot.LR)
        )
        for slot_idx in range(lr.count)
    )
    lr_opcode_fields = tuple(opcode_field(Slot.LR, instance) for instance in range(lr.count))
    return opcode_fields, plans, lr_plans, lr_opcode_fields


# ---------------------------------------------------------------------------
# IPU Execution Engine
# ---------------------------------------------------------------------------


class Ipu:
    """IPU execution engine with automatic instruction dispatch.

    Orchestration only: the engine holds the state, the cycle-start snapshot
    binding, the :class:`~ipu_emu.cycle.CycleView` handlers read through, the
    issuer and the cycle. The handlers are the functions of
    :mod:`ipu_emu.slots`, called with this engine as their first argument.

    Attributes:
        state: IPU state (regfile, xmem, program counter, etc.)
        snapshot: Register file snapshot for VLIW read-before-write semantics
    """

    def __init__(self, state: IpuState):
        """Initialize IPU execution engine.

        Args:
            state: IPU state containing regfile, xmem, instruction memory, etc.
        """
        self.state = state
        # Prototyping mode's shortcuts, read once: the flag belongs to the run,
        # not to the cycle (see ipu_emu/prototyping.py).
        self._prototyping: bool = state.prototyping
        bridge = None
        if state.alias_profile is not None:
            profile = state.alias_profile
            bridge = profile.bind(state)
            state.stats.alias_profile = profile
        # The cycle's register snapshot. The Cycle owns it (bind_snapshot);
        # the issuer resolves cycle-start operands from it, and handlers read
        # it only through ``view``.
        self._snapshot_buffer: RegFile | None = None
        # Every register read a handler makes goes through this view, which
        # serves it from the snapshot or the live file as the instruction
        # declares (E04-12). The issuer sets its plan before each op.
        self._view = CycleView(state)
        # Issues each slot op into one IssueFacts record and runs its handler.
        # The record of the op in flight is ``issuer.current``: the multiply
        # handlers read its identity verdict when they count lanes.
        # How its read operands become register values: the resolver adapter
        # the cycle module picks for this state (packed when prototyping).
        self.issuer = Issuer(self, resolver_for(state))
        # One VLIW word's sequencing: snapshot, slot order, idle words, the LR
        # conflict check and the observers' hooks. The cycle gets a
        # weak proxy: a strong back-reference would make every engine a
        # reference cycle, keeping its state's XMEM alive until the cyclic
        # garbage collector happens to run.
        # So ``cycle`` works only while this Ipu is alive: hold the Ipu, or
        # build a Cycle of your own (``Cycle(state, ipu)``).
        self.cycle = Cycle(state, weakref.proxy(self))
        if bridge is not None:
            # The profiler observes this engine's cycles, and reads the snapshot
            # through this cycle (Cycle.snapshot).
            self.cycle.add_observer(bridge.observing(self.cycle))

    @property
    def snapshot(self) -> RegFile | None:
        """Inspectable, cycle-stable copy of the current execution snapshot.

        See :attr:`Cycle.snapshot`.
        """
        return self.cycle.snapshot

    @property
    def view(self) -> CycleView:
        """The view every handler reads registers through: the cycle-start
        snapshot or the live file, as the instruction in flight declares."""
        return self._view

    @property
    def snapshot_buffer(self) -> RegFile | None:
        """The cycle's register snapshot itself (not a copy; see :attr:`snapshot`).
        The issuer resolves cycle-start operands from it."""
        return self._snapshot_buffer

    @property
    def prototyping(self) -> bool:
        """Whether this engine takes prototyping mode's shortcuts (read once,
        when the engine is built)."""
        return self._prototyping

    @property
    def codec(self) -> LaneCodec:
        """The state's lane codec (``IpuState.codec``, of ``IpuState.fmt``).

        The handlers decode, compute and store through it, so none branches on
        a format (Q43, Q48).
        """
        return self.state.codec

    def bind_snapshot(self, buffer: RegFile) -> None:
        """Make *buffer* the register snapshot handlers read. Called by :class:`Cycle`.

        Args:
            buffer: The snapshot the cycle refreshes at each cycle start.
        """
        self._snapshot_buffer = buffer
        self._view.snapshot = buffer


_SLOT_OPCODE_FIELDS, _SLOT_PLANS, _LR_PLANS, _LR_OPCODE_FIELDS = _build_slot_plans()


def plans_for(slot: Slot) -> tuple[tuple[str, tuple[SlotPlan, ...]], ...]:
    """The decode tables of one slot, for :mod:`ipu_emu.cycle`.

    Args:
        slot: The slot.

    Returns:
        ``(opcode field, plans indexed by opcode)`` per issue position: one for
        most slots, one per sub-slot for LR.
    """
    if slot is Slot.LR:
        return tuple(zip(_LR_OPCODE_FIELDS, _LR_PLANS, strict=True))
    return ((_SLOT_OPCODE_FIELDS[slot], _SLOT_PLANS[slot]),)


def _check_execution_order_covers_plans() -> None:
    """Every slot with plans is in ``SLOT_EXECUTION_ORDER``, and only those.

    The cycle loops iterate the declared order, so a slot missing from it
    would silently never execute. Raised, not asserted, so ``-O`` keeps it.
    """
    planned = {Slot(slot) for slot in _SLOT_PLANS} | {Slot.LR}
    declared = set(SLOT_EXECUTION_ORDER)
    if declared != planned:
        raise ConfigurationError(
            "ipu_isa.timing.SLOT_EXECUTION_ORDER must name exactly the slots the "
            f"emulator has plans for; missing {sorted(map(str, planned - declared))}, "
            f"unknown {sorted(map(str, declared - planned))}"
        )


_check_execution_order_covers_plans()


# ``Ipu`` builds its ``Cycle``, and the cycle module reads this module's decode
# tables (``plans_for``), so the two modules depend on each other. Each imports
# the other last, so either can be imported first (see the end of
# ``ipu_emu.cycle``). Removing the cycle means moving the plan tables to a
# module of their own.
from ipu_emu.cycle import Cycle, CycleView, resolver_for  # noqa: E402
