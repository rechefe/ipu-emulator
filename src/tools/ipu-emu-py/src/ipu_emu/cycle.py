"""One VLIW word, one implementation: the :class:`Cycle` module.

A cycle fetches the word at the PC and does what that word does: BREAK is
evaluated against a fresh register snapshot before any side effect, then every
other slot runs in ``ipu_isa.timing.SLOT_EXECUTION_ORDER``. An empty word (no
instruction at the PC) is an idle cycle: it counts one idle multiplier cycle and
advances the PC. Two LR sub-slots writing the same register is an error raised
before any LR write.

:class:`Cycle` owns all of that: the snapshot, the slot order, the idle-word
path, the LR conflict check and the observer's begin/end hooks. It knows nothing
about run control: cycle limits, stats totals and debug callbacks belong to its
callers (``ipu_emu.emulator`` today, the steppable engine later).

The break-then-resume protocol a debugger needs is two calls:
:meth:`Cycle.evaluate_break` stops before any side effect, and
:meth:`Cycle.retire` later runs everything except BREAK. :meth:`Cycle.step` is
the two together.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Protocol

from ipu_isa.registers import lrd_pair
from ipu_isa.slots import Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu, SlotPlan
    from ipu_emu.ipu_state import IpuState
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
            mutated: a caller that resolves reads copies it first.
    """

    slot: str
    plan: SlotPlan
    operands: Mapping[str, int]


class Schedule(NamedTuple):
    """What one instruction word does, in issue order. A pure function of the word.

    Immutable (a named tuple: built once per cycle, so cheaper than a frozen
    dataclass).

    Attributes:
        break_op: The BREAK slot, or ``None`` when it holds a NOP.
        lr_ops: Every LR sub-slot, in sub-slot order.
        slot_ops: Every other slot except BREAK, in ``SLOT_EXECUTION_ORDER``.
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
    """

    def begin(self, pc: int) -> None: ...

    def end(self) -> None: ...


def lr_conflict_error(targets: tuple[int, ...]) -> RuntimeError:
    """The error for two LR sub-slots writing the same register.

    Args:
        targets: ``Schedule.lr_conflict`` of the word.

    Returns:
        The exception to raise, before any LR write.
    """
    return RuntimeError(
        "LR conflict: multiple writes to the same LR register in same cycle "
        f"(targets: {list(targets)})"
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
        plan = plans[inst[field]]
        ops.append(IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields}))
    return tuple(ops)


def schedule_word(inst: Mapping[str, int]) -> Schedule:
    """Build the :class:`Schedule` of one instruction word.

    Args:
        inst: A decoded instruction word (field name to value).

    Returns:
        The word's schedule, with its LR conflict checked once.
    """
    break_op = None
    lr_ops: list[IssuedOp] = []
    slot_ops: list[IssuedOp] = []
    lr_position = -1
    targets: list[int] = []
    for slot, field, plans in _issue_table(tuple(SLOT_EXECUTION_ORDER)):
        plan = plans[inst[field]]
        if slot == "break":
            if plan.inst_name != "NOP":
                break_op = IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields})
            continue
        op = IssuedOp(slot, plan, {name: inst[key] for name, key in plan.fields})
        if slot == "lr":
            if lr_position < 0:
                lr_position = len(slot_ops)
            lr_ops.append(op)
            if plan.write_operand is not None:
                raw = op.operands[plan.write_operand]
                # An LrdIdx target writes the two LRs of the pair.
                if plan.write_is_lrd:
                    targets.extend(lrd_pair(raw))
                else:
                    targets.append(raw)
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

    Args:
        state: The machine state the cycle reads and writes.
        ipu: The handlers and slot dispatch the cycle drives. Held as given:
            ``Ipu`` passes a weak proxy of itself, so it is not a reference cycle.
        observer: Optional begin/end hooks (the alias profiler today).
    """

    def __init__(self, state: IpuState, ipu: Ipu, observer: CycleObserver | None = None) -> None:
        self.state = state
        self._ipu = ipu
        self._observer = observer
        # Prototyping mode's NOP-skipping cycle, read once: the flag belongs to
        # the run, not to the cycle. A profiled run takes the faithful path so
        # the profile describes every slot the cycle issued.
        self._fast = state.prototyping and observer is None
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

    def evaluate_break(self) -> CycleOutcome:
        """Take the snapshot and evaluate the BREAK slot only. No side effects.

        Returns:
            ``BREAK`` if the word's BREAK slot fires, else ``RETIRED`` (the word
            is still to be retired with :meth:`retire`).
        """
        inst = self.state.inst_mem[self.state.program_counter]
        if inst is None:
            return CycleOutcome.RETIRED
        self._take_snapshot()
        return self._evaluate_break(inst, schedule_word(inst))

    def retire(self) -> None:
        """Take a fresh snapshot, then run every slot except BREAK, in order.

        The snapshot is taken again on purpose: an edit a debugger made while
        stopped at a break is visible to the resumed word.
        """
        pc = self.state.program_counter
        inst = self.state.inst_mem[pc]
        if inst is None:
            self._run(pc, None, None)
            return
        self._take_snapshot()
        if self._fast:
            self._ipu._execute_cycle_fast(inst, pc, with_break=False)
            return
        self._run(pc, inst, schedule_word(inst))

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
        inst = self.state.inst_mem[pc]
        if inst is None:
            self._run(pc, None, None)
            return CycleOutcome.RETIRED
        self._take_snapshot()
        if self._fast:
            return self._ipu._execute_cycle_fast(inst, pc, with_break=honour_break)
        schedule = schedule_word(inst)
        if honour_break and self._evaluate_break(inst, schedule) is CycleOutcome.BREAK:
            return CycleOutcome.BREAK
        self._run(pc, inst, schedule)
        return CycleOutcome.RETIRED

    def reset(self) -> None:
        """Forget cached schedules and the snapshot source.

        The next cycle copies the register file afresh, and no schedule cached
        for an old program survives.
        """
        self._source = None
        self._ipu.forget_cycle_plans()

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
        return self._ipu.dispatch_op(schedule.break_op, inst)

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
        dispatch_op = self._ipu.dispatch_op
        slot_ops = schedule.slot_ops
        lr_position = schedule.lr_position
        for op in slot_ops[:lr_position]:
            dispatch_op(op, inst)
        if schedule.lr_conflict is not None:
            raise lr_conflict_error(schedule.lr_conflict)
        self._ipu.dispatch_lr(schedule.lr_ops, inst)
        for op in slot_ops[lr_position:]:
            dispatch_op(op, inst)


# ``ipu_emu.ipu`` holds the decode tables (a plan holds its handler function) and
# builds its ``Cycle`` from this module. Each module imports the other last, so
# either can be imported first. The Ipu-side edge goes once callers stop
# entering through Ipu's cycle delegates (E04-11, E08).
from ipu_emu.ipu import plans_for  # noqa: E402
