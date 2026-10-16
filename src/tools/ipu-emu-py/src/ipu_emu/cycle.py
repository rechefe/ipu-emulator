"""One VLIW word, one implementation: the :class:`Cycle` module.

A cycle fetches the word at the PC and does what that word does: BREAK is
evaluated against a fresh register snapshot before any side effect, then every
other slot runs in ``ipu_isa.timing.SLOT_EXECUTION_ORDER``. An empty word (no
instruction at the PC) is an idle cycle: it counts one idle multiplier cycle and
advances the PC. Two LR sub-slots writing the same register is fault F-3,
raised before any LR write.

An illegal program faults (E04-19): a fetch at or past ``program_length`` is
F-2, an undefined opcode or a reserved operand code (or one past its type's
width) F-1, an XMEM access beyond the allocation F-6, and
handlers raise the rest. Every :class:`EmulatorFault` leaving a cycle has first
recorded ``IpuState.halt``.

:class:`Cycle` owns all of that: the snapshot, the slot order, the idle-word
path, the LR conflict check and the observer's begin/end hooks. It knows nothing
about run control: cycle limits, stats totals and debug callbacks belong to its
callers (``ipu_emu.emulator`` today, the steppable engine later).

The break-then-resume protocol a debugger needs is two calls:
:meth:`Cycle.evaluate_break` stops before any side effect, and
:meth:`Cycle.retire` later runs everything except BREAK. :meth:`Cycle.step` is
the two together.

What a word does is a :class:`Schedule`, built once per PC by
:func:`schedule_word` and cached until that IMEM entry is written (E04-08).
Prototyping mode is an adapter on this one cycle: its schedules skip the idle
NOP slots (``skip_idle_nops=True``) and nothing else differs.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Protocol

from ipu_emu.errors import EmulatorFault, XMemBoundsError
from ipu_emu.ipu_state import HaltRecord
from ipu_isa.faults import FaultId
from ipu_isa.registers import lrd_pair
from ipu_isa.slots import Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu, SlotPlan
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.issue import IssueFacts
    from ipu_emu.regfile import RegFile


class CycleOutcome(IntEnum):
    """How one call into the cycle ended.

    ``CONTINUE`` is an alias of ``RETIRED``, so code written against the old
    ``BreakResult`` (an alias of this class until E08 migrates the debuggers)
    keeps working.
    """

    RETIRED = 0
    BREAK = 1
    CONTINUE = 0


class IssuedOp(NamedTuple):
    """One slot of an instruction word, decoded.

    Attributes:
        slot: The slot type (``"lr"`` for every LR sub-slot).
        plan: What dispatch needs for this (slot, opcode).
        operands: The raw operand values the word holds, by operand name. Never
            mutated: a caller that resolves reads copies it first. ``None``
            marks a skipped NOP that only bumps its counter (``plan.stat``):
            prototyping's schedules keep such a slot in place instead of
            dispatching it.
    """

    slot: str
    plan: SlotPlan
    operands: Mapping[str, int] | None


class Schedule(NamedTuple):
    """What one instruction word does, in issue order. A pure function of the word.

    Immutable (a named tuple: built once per cycle, so cheaper than a frozen
    dataclass).

    Attributes:
        break_op: The BREAK slot, or ``None`` when it holds a NOP.
        lr_ops: Every LR sub-slot, in sub-slot order (without the NOPs when
            idle NOPs are skipped).
        slot_ops: Every other slot except BREAK, in ``SLOT_EXECUTION_ORDER``.
            When idle NOPs are skipped, a NOP whose handler does nothing is
            left out, or kept as a counter-only op if it bumps a counter.
        lr_position: Where in ``slot_ops`` the LR sub-slots run: they run before
            ``slot_ops[lr_position]`` (after the last op when it equals the length).
        lr_conflict: The LR registers the sub-slots write, in sub-slot order,
            when two of them are the same register; ``None`` otherwise.
    """

    break_op: IssuedOp | None
    lr_ops: tuple[IssuedOp, ...]
    slot_ops: tuple[IssuedOp, ...]
    lr_position: int
    lr_conflict: tuple[int, ...] | None


class CycleObserver(Protocol):
    """Hooks around the part of a cycle that has side effects.

    ``begin`` runs before the first slot after BREAK (or before an idle word's
    PC advance), ``end`` after the last. A cycle stopped by BREAK calls neither.
    Both modes call it the same way: a schedule that skips idle NOPs still
    brackets the same words.
    """

    def begin(self, pc: int) -> None: ...

    def end(self) -> None: ...


def lr_conflict_error(targets: tuple[int, ...], pc: int) -> EmulatorFault:
    """Fault F-3: two LR sub-slots write the same register.

    Args:
        targets: ``Schedule.lr_conflict`` of the word.
        pc: The word's PC.

    Returns:
        The fault to raise, before any LR write.
    """
    return EmulatorFault(
        FaultId.F3,
        pc,
        "LR conflict: multiple writes to the same LR register in same cycle "
        f"(targets: {list(targets)})",
    )


@lru_cache(maxsize=4)
def _issue_table(order: tuple[Slot, ...]) -> tuple[tuple[str, str, tuple[SlotPlan, ...]], ...]:
    """(slot type, opcode field, opcode-indexed plans) per issue position of *order*."""
    return tuple((slot.value, field, plans) for slot in order for field, plans in plans_for(slot))


def decode_word(inst: Mapping[str, int]) -> tuple[IssuedOp, ...]:
    """Decode every slot of an instruction word, NOPs included.

    Args:
        inst: A decoded instruction word (field name to value).

    Returns:
        One op per slot in ``SLOT_EXECUTION_ORDER``; the LR slot gives one op
        per sub-slot.
    """
    # The order is read from the module global on every call, so the ISA
    # package stays the one place it is declared (tests patch it here).
    ops = []
    for slot, field, plans in _issue_table(tuple(SLOT_EXECUTION_ORDER)):
        plan = _plan(slot, field, plans, inst, -1)
        ops.append(IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields}))
    return tuple(ops)


def _plan(
    slot: str, field: str, plans: tuple[SlotPlan, ...], inst: Mapping[str, int], pc: int
) -> SlotPlan:
    """The plan of the opcode in ``inst[field]``; F-1 when the ISA defines none."""
    code = inst[field]
    if not 0 <= code < len(plans):
        raise EmulatorFault(
            FaultId.F1, pc, f"{slot} slot: opcode {code} ({field}) is not defined by the ISA"
        )
    return plans[code]


def _check_operands(slot: str, plan: SlotPlan, inst: Mapping[str, int], pc: int) -> None:
    """F-1 for an operand code past its type's width, or reserved."""
    fields = dict(plan.fields)
    for name, capacity, reserved in plan.operand_checks:
        code = inst[fields[name]]
        if not 0 <= code < capacity or code in reserved:
            raise EmulatorFault(
                FaultId.F1,
                pc,
                f"{slot} slot: {plan.inst_name} {name} = {code} is a reserved encoding",
            )


def _lr_targets(plan: SlotPlan, operands: Mapping[str, int]) -> tuple[int, ...]:
    """The LRs one LR sub-slot writes: none, one, or an LrdIdx target's pair."""
    if plan.write_operand is None:
        return ()
    raw = operands[plan.write_operand]
    return tuple(lrd_pair(raw)) if plan.write_is_lrd else (raw,)


#: Slots whose NOP handler is not a no-op: the cond slot's NOP advances the PC,
#: so a schedule that skips idle NOPs still issues it.
_ACTIVE_NOP_SLOTS = frozenset({Slot.COND.value})


def schedule_word(
    inst: Mapping[str, int], *, skip_idle_nops: bool = False, pc: int = -1
) -> Schedule:
    """Build the :class:`Schedule` of one instruction word.

    Args:
        inst: A decoded instruction word (field name to value).
        skip_idle_nops: Leave out the NOP slots whose handler does nothing
            (prototyping mode). A skipped NOP that bumps a counter (an idle
            multiplier counts ``mult_idle_cycles``) stays in place as a
            counter-only op, so a later slot's exception truncates the cycle
            where the faithful schedule would. The cond slot's NOP is never
            skipped: it advances the PC.
        pc: The word's PC, for the F-1 fault the word may raise.

    Returns:
        The word's schedule, with its LR conflict checked once.

    Raises:
        EmulatorFault: F-1, a slot holds an opcode the ISA does not define,
            or an operand code that is reserved or past its type's width.
    """
    break_op = None
    lr_ops: list[IssuedOp] = []
    slot_ops: list[IssuedOp] = []
    lr_position = -1
    targets: list[int] = []
    for slot, field, plans in _issue_table(tuple(SLOT_EXECUTION_ORDER)):
        plan = _plan(slot, field, plans, inst, pc)
        _check_operands(slot, plan, inst, pc)
        if slot == "break":
            if plan.inst_name != "NOP":
                break_op = IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields})
            continue
        if slot == "lr" and lr_position < 0:
            lr_position = len(slot_ops)
        if skip_idle_nops and plan.inst_name == "NOP" and slot not in _ACTIVE_NOP_SLOTS:
            if plan.stat is not None:
                # Kept in place with no operands: issued for its counter alone.
                slot_ops.append(IssuedOp(slot, plan, None))
            continue
        op = IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields})
        if slot == "lr":
            lr_ops.append(op)
            targets.extend(_lr_targets(plan, op.operands))
        else:
            slot_ops.append(op)
    if lr_position < 0:  # unreachable while the declared order names LR
        lr_position = len(slot_ops)
    return Schedule(
        break_op=break_op,
        lr_ops=tuple(lr_ops),
        slot_ops=tuple(slot_ops),
        lr_position=lr_position,
        lr_conflict=tuple(targets) if len(targets) != len(set(targets)) else None,
    )


class Cycle:
    """Executes the instruction word at the PC, one cycle per call.

    Each PC's :class:`Schedule` is built on first use and cached until that IMEM
    entry is written; :meth:`reset` drops them all.

    Args:
        state: The machine state the cycle reads and writes.
        ipu: The handlers and slot dispatch the cycle drives. Held as given:
            ``Ipu`` passes a weak proxy of itself, so it is not a reference cycle.
        observer: Optional begin/end hooks (the alias profiler today).
        skip_idle_nops: Build schedules that skip idle NOP slots. Defaults to
            ``state.prototyping``: prototyping mode is this flag (and its
            operand resolver) and nothing else. Results are identical either
            way, so a test may pass it in any build.
    """

    def __init__(
        self,
        state: IpuState,
        ipu: Ipu,
        observer: CycleObserver | None = None,
        *,
        skip_idle_nops: bool | None = None,
    ) -> None:
        self.state = state
        self._ipu = ipu
        self._observer = observer
        # Read once: the flag belongs to the run, not to the cycle.
        self._skip_idle_nops = state.prototyping if skip_idle_nops is None else skip_idle_nops
        # PC -> (the word, its schedule); (None, None) for an empty entry. The
        # word is IMEM's frozen copy, so the entry stays right until that PC
        # is written, and the IMEM write observer drops it then.
        self._schedules: dict[int, tuple[Mapping[str, int] | None, Schedule | None]] = {}
        schedules = self._schedules
        remove = state.inst_mem.add_write_observer(lambda pc: schedules.pop(pc, None))
        # The observer holds the dict, not the cycle; it is removed when the
        # cycle is collected, so a state outliving many engines does not
        # collect dead observers.
        weakref.finalize(self, remove)
        # The snapshot buffer is reused across cycles: every register is
        # refreshed from the live file at cycle start, so it holds exactly what
        # a fresh copy would. ``_source`` is the file it copies.
        self._buffer: RegFile | None = None
        self._source: RegFile | None = None
        # Public snapshot, materialized only when inspected and never recycled,
        # so a debugger may retain it across later cycles.
        self._public_snapshot: RegFile | None = None

    @property
    def snapshot(self) -> RegFile | None:
        """Inspectable, cycle-stable copy of this cycle's register snapshot.

        Handlers read the reused buffer, so normal execution does not allocate
        here. A debugger pays for one copy only if it inspects this property;
        repeated reads in one cycle return the same object. ``None`` before the
        first cycle.
        """
        if self._buffer is None:
            return None
        if self._public_snapshot is None:
            self._public_snapshot = self._buffer.snapshot()
        return self._public_snapshot

    @property
    def current(self) -> IssueFacts | None:
        """The :class:`~ipu_emu.issue.IssueFacts` of the op whose handler is running.

        Kept by the engine's issuer, so a handler reads the same record whichever
        cycle drives it. The last op's record between ops; ``None`` before the
        first.
        """
        return self._ipu.issuer.current

    @property
    def skip_idle_nops(self) -> bool:
        """Whether this cycle's schedules skip idle NOP slots (prototyping)."""
        return self._skip_idle_nops

    def cached_schedule(self, pc: int) -> Schedule | None:
        """The schedule cached for *pc*, or ``None`` if none is (or *pc* is empty).

        Args:
            pc: An IMEM address.

        Returns:
            What the next cycle at *pc* would reuse without decoding the word.
        """
        entry = self._schedules.get(pc)
        return None if entry is None else entry[1]

    def evaluate_break(self) -> CycleOutcome:
        """Take the snapshot and evaluate the BREAK slot only. No side effects.

        Returns:
            ``BREAK`` if the word's BREAK slot fires, else ``RETIRED`` (the word
            is still to be retired with :meth:`retire`).
        """
        pc = self.state.program_counter
        try:
            inst, schedule = self._fetch(pc)
            if schedule is None:
                return CycleOutcome.RETIRED
            self._take_snapshot()
            return self._evaluate_break(inst, schedule)
        except (EmulatorFault, XMemBoundsError) as error:
            raise self._halt_on(error, pc) from None

    def retire(self) -> None:
        """Take a fresh snapshot, then run every slot except BREAK, in order.

        The snapshot is taken again on purpose: an edit a debugger made while
        stopped at a break is visible to the resumed word, and so is a word it
        wrote into IMEM (that write dropped the cached schedule).
        """
        pc = self.state.program_counter
        try:
            inst, schedule = self._fetch(pc)
            if schedule is not None:
                self._take_snapshot()
            self._run(pc, inst, schedule)
        except (EmulatorFault, XMemBoundsError) as error:
            raise self._halt_on(error, pc) from None

    def step(self, *, honour_break: bool = True) -> CycleOutcome:
        """Execute one word: :meth:`evaluate_break`, then :meth:`retire`.

        Takes the snapshot once. The word's effects equal the two calls made in
        turn with no edit in between.

        Args:
            honour_break: When false, BREAK is not evaluated and the word
                retires.

        Returns:
            ``BREAK`` if the word stopped at its BREAK slot (nothing retired),
            else ``RETIRED``.
        """
        pc = self.state.program_counter
        try:
            entry = self._schedules.get(pc) if pc < self.state.program_length else None
            inst, schedule = self._fetch(pc) if entry is None else entry
            if schedule is not None:
                self._take_snapshot()
                if (
                    honour_break
                    and schedule.break_op is not None
                    and self._ipu.issuer.dispatch(schedule.break_op, inst) is CycleOutcome.BREAK
                ):
                    return CycleOutcome.BREAK
            self._run(pc, inst, schedule)
        except (EmulatorFault, XMemBoundsError) as error:
            raise self._halt_on(error, pc) from None
        return CycleOutcome.RETIRED

    def reset(self) -> None:
        """Forget cached schedules and the snapshot source.

        The next cycle copies the register file afresh, and no schedule cached
        for an old program survives.
        """
        self._source = None
        self._schedules.clear()

    def _halt_on(self, error: EmulatorFault | XMemBoundsError, pc: int) -> EmulatorFault:
        """Record the halt for a fault in the word at *pc*; an XMEM access
        beyond the allocation becomes F-6 there."""
        fault = (
            error if isinstance(error, EmulatorFault) else EmulatorFault(FaultId.F6, pc, str(error))
        )
        self.state.halt = HaltRecord("fault", fault.pc, fault.fault)
        return fault

    def _fetch(self, pc: int) -> tuple[Mapping[str, int] | None, Schedule | None]:
        """The word at *pc* and its schedule, from the cache or built now.

        Raises:
            EmulatorFault: F-2, *pc* is outside the loaded program; F-1, the
                word holds an undefined opcode or a reserved operand code.
        """
        if not 0 <= pc < self.state.program_length:
            raise EmulatorFault(
                FaultId.F2,
                pc,
                f"fetch outside the program (length {self.state.program_length})",
            )
        entry = self._schedules.get(pc)
        if entry is not None:
            return entry
        inst = self.state.inst_mem[pc]
        entry = (
            inst,
            None
            if inst is None
            else schedule_word(inst, skip_idle_nops=self._skip_idle_nops, pc=pc),
        )
        self._schedules[pc] = entry
        return entry

    def _take_snapshot(self) -> None:
        regfile = self.state.regfile
        if regfile is not self._source or self._buffer is None:
            # First cycle, after reset, or the state swapped in another
            # register file: take a real copy of this one and reuse it.
            self._buffer = regfile.snapshot()
            self._source = regfile
            self._ipu.bind_snapshot(self._buffer)
        else:
            regfile.copy_into(self._buffer)
        # A previously exposed snapshot stays frozen for any debugger retaining
        # it. The next public access lazily copies this cycle.
        self._public_snapshot = None

    def _evaluate_break(self, inst: Mapping[str, int], schedule: Schedule) -> CycleOutcome:
        if schedule.break_op is None:
            return CycleOutcome.RETIRED
        return self._ipu.issuer.dispatch(schedule.break_op, inst)

    def _run(self, pc: int, inst: Mapping[str, int] | None, schedule: Schedule | None) -> None:
        """Every side effect of one word, between the observer's begin and end."""
        observer = self._observer
        if observer is not None:
            observer.begin(pc)
        if inst is None or schedule is None:
            # An idle word: the multiplier sits idle and the PC advances.
            self.state.stats.mult_idle_cycles += 1
            self.state.program_counter += 1
        else:
            self._run_slots(inst, schedule)
        if observer is not None:
            observer.end()

    def _run_slots(self, inst: Mapping[str, int], schedule: Schedule) -> None:
        issuer = self._ipu.issuer
        dispatch = issuer.dispatch
        stats = self.state.stats
        slot_ops = schedule.slot_ops
        lr_position = schedule.lr_position
        for op in slot_ops[:lr_position]:
            if op.operands is None:  # a skipped NOP, issued for its counter
                setattr(stats, op.plan.stat, getattr(stats, op.plan.stat) + 1)
            else:
                dispatch(op, inst)
        if schedule.lr_conflict is not None:
            raise lr_conflict_error(schedule.lr_conflict, self.state.program_counter)
        # Every LR sub-slot resolves its reads before any of them writes, in
        # both modes, so a read that raises leaves no LR written.
        issuer.dispatch_all(schedule.lr_ops, inst)
        for op in slot_ops[lr_position:]:
            if op.operands is None:
                setattr(stats, op.plan.stat, getattr(stats, op.plan.stat) + 1)
            else:
                dispatch(op, inst)


# ``ipu_emu.ipu`` holds the decode tables (a plan holds its handler function) and
# builds its ``Cycle`` from this module. Each module imports the other last, so
# either can be imported first. The Ipu-side edge goes once callers stop
# entering through Ipu's cycle delegates (E04-11, E08).
from ipu_emu.ipu import plans_for  # noqa: E402
