"""One VLIW word, one implementation: the :class:`Cycle` module.

A cycle fetches the word at the PC and does what that word does: BREAK is
evaluated against a fresh register snapshot before any side effect, then every
other slot runs in ``ipu_isa.timing.SLOT_EXECUTION_ORDER``. An empty word (no
instruction at the PC) is an idle cycle: it counts one idle multiplier cycle and
advances the PC. Two LR sub-slots writing the same register is an error raised
before any LR write.

:class:`Cycle` owns all of that: the snapshot, the slot order, the idle-word
path, the LR conflict check and the observers' hooks. It knows nothing
about run control: cycle limits, stats totals and debug callbacks belong to its
callers (``ipu_emu.emulator`` today, the steppable engine later).

The break-then-resume protocol a debugger needs is two calls:
:meth:`Cycle.evaluate_break` stops before any side effect, and
:meth:`Cycle.retire` later runs everything except BREAK. :meth:`Cycle.step` is
the two together.

What a word does is a :class:`Schedule`, built once per PC by
:func:`schedule_word` and cached until that IMEM entry is written (E04-08).
Prototyping mode is an adapter on this one cycle: its schedules skip the idle
NOP slots (``skip_idle_nops=True``) and its reads go through the packed operand
resolver (:func:`resolver_for`); nothing else differs.

Handlers read registers through :class:`CycleView`, which serves each read from
the snapshot or the live register file as the instruction's timing
declaration says (E04-12).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from ipu_emu.errors import EmulatorError, UndeclaredRead
from ipu_emu.outcome import CycleOutcome
from ipu_emu.resolver import FaithfulResolver, OperandResolver, PackedResolver
from ipu_isa.constants import DStructureConfig, decode_dstructure
from ipu_isa.registers import lrd_pair
from ipu_isa.slots import Slot
from ipu_isa.timing import SLOT_EXECUTION_ORDER

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu, SlotPlan
    from ipu_emu.ipu_state import IpuState
    from ipu_emu.issue import IssueFacts
    from ipu_emu.regfile import RegFile, RegFileReader


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
        lr_writes: The LRs an LR sub-slot writes (none, one, or an LRD pair),
            from the word's schedule; empty for every other slot.
    """

    slot: str
    plan: SlotPlan
    operands: Mapping[str, int] | None
    lr_writes: tuple[int, ...] = ()


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
    """What an observer of the cycle sees: public fact records, nothing else.

    ``begin`` runs before the first slot after BREAK (or before an idle word's
    PC advance), ``end`` after the last. A cycle stopped by BREAK calls
    neither. ``issued`` runs just before an op's handler, with the op's
    :class:`~ipu_emu.issue.IssueFacts`, and ``retired`` just after it, with
    the handler's result; an op whose handler raises is never retired. The
    BREAK slot's op is issued and retired before ``begin`` (and, when it
    fires, with no ``begin`` at all).

    Both modes call it the same way and bracket the same words, but a
    schedule that skips idle NOPs issues no op for a skipped NOP slot: an
    observer that counts ops ignores NOPs to count the same in both.

    Observers are attached with :meth:`Cycle.add_observer` (one cycle) or
    :meth:`~ipu_emu.ipu_state.IpuState.add_cycle_observer` (every cycle that
    drives the state).
    """

    def begin(self, pc: int) -> None: ...

    def issued(self, facts: IssueFacts) -> None: ...

    def retired(self, facts: IssueFacts, result: Any) -> None: ...

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
        operands = {name: inst[key] for name, key in plan.fields}
        writes = _lr_targets(plan, operands) if slot == "lr" else ()
        ops.append(IssuedOp(slot, plan, operands, writes))
    return tuple(ops)


def _lr_targets(plan: SlotPlan, operands: Mapping[str, int]) -> tuple[int, ...]:
    """The LRs one LR sub-slot writes: none, one, or an LrdIdx target's pair."""
    if plan.write_operand is None:
        return ()
    raw = operands[plan.write_operand]
    return tuple(lrd_pair(raw)) if plan.write_is_lrd else (raw,)


#: Slots whose NOP handler is not a no-op: the cond slot's NOP advances the PC,
#: so a schedule that skips idle NOPs still issues it.
_ACTIVE_NOP_SLOTS = frozenset({Slot.COND.value})


def schedule_word(inst: Mapping[str, int], *, skip_idle_nops: bool = False) -> Schedule:
    """Build the :class:`Schedule` of one instruction word.

    Args:
        inst: A decoded instruction word (field name to value).
        skip_idle_nops: Leave out the NOP slots whose handler does nothing
            (prototyping mode). A skipped NOP that bumps a counter (an idle
            multiplier counts ``mult_idle_cycles``) stays in place as a
            counter-only op, so a later slot's exception truncates the cycle
            where the faithful schedule would. The cond slot's NOP is never
            skipped: it advances the PC.

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
        if slot == "lr" and lr_position < 0:
            lr_position = len(slot_ops)
        if skip_idle_nops and plan.inst_name == "NOP" and slot not in _ACTIVE_NOP_SLOTS:
            if plan.stat is not None:
                # Kept in place with no operands: issued for its counter alone.
                slot_ops.append(IssuedOp(slot, plan, None))
            continue
        operands = {name: inst[key] for name, key in plan.fields}
        if slot == "lr":
            writes = _lr_targets(plan, operands)
            lr_ops.append(IssuedOp(slot, plan, operands, writes))
            targets.extend(writes)
        else:
            slot_ops.append(IssuedOp(slot, plan, operands))
    if lr_position < 0:  # unreachable while the declared order names LR
        lr_position = len(slot_ops)
    return Schedule(
        break_op=break_op,
        lr_ops=tuple(lr_ops),
        slot_ops=tuple(slot_ops),
        lr_position=lr_position,
        lr_conflict=tuple(targets) if len(targets) != len(set(targets)) else None,
    )


#: Emulator-only storage that holds a declared register in wide-vector debug
#: mode: a read of it has the view declared for that register.
_DECLARED_AS = {"r_wide_debug": "r", "r_cyclic_wide_debug": "r_cyclic"}


class CycleView:
    """How a handler reads registers: through the view its instruction declares.

    The ISA timing contract (``ipu_isa.timing``, Q32) declares, for every
    register an instruction reads, which value the read sees: the cycle-start
    snapshot or the forwarded (live) one. The declaration is the instruction's
    ``read`` operands and ``implicit_reads``; :attr:`SlotPlan.views
    <ipu_emu.ipu.SlotPlan>` holds it per plan. A handler names the register
    (and, where it matters, the operand that selects it) and this class picks
    the register file, so changing when a read happens is an edit to the spec,
    not to a handler.

    The issuer sets :attr:`plan` before it issues each op and again before its
    handler runs; :class:`Cycle` sets :attr:`snapshot` when it takes the
    cycle's snapshot. Writes never go through here: handlers write the live
    file (``RegFile.mutable`` and the ``set_*`` accessors).

    Args:
        state: The machine state whose live register file is read.
    """

    __slots__ = ("_snapshot", "_state", "plan")

    def __init__(self, state: IpuState) -> None:
        self._state = state
        self._snapshot: RegFile | None = None
        #: The plan of the op being issued or executed; ``None`` before the first.
        self.plan: SlotPlan | None = None

    @property
    def snapshot(self) -> RegFile:
        """The cycle-start register file (read only by handlers).

        Raises:
            EmulatorError: No cycle has taken a snapshot yet.
        """
        snapshot = self._snapshot
        if snapshot is None:
            raise EmulatorError("no register snapshot: a handler ran outside a cycle")
        return snapshot

    @snapshot.setter
    def snapshot(self, buffer: RegFile) -> None:
        self._snapshot = buffer

    @property
    def live(self) -> RegFile:
        """The register file this cycle writes (the forwarded view)."""
        return self._state.regfile

    def source(self, register: str, via: str | None = None) -> RegFile:
        """The register file a read of *register* comes from: snapshot or live.

        Args:
            register: A ``REGISTER_DEFINITIONS`` name (``"r_mask"``, ``"lr"``,
                ...); a wide-vector debug register reads as the register it
                stands in for.
            via: The operand that selects the register's index, as declared
                (``"src"``, ``"reshape_mask"``); ``None`` when the
                instruction's declared reads of *register* all share one view.

        Returns:
            The cycle-start snapshot or the live register file. Only its read
            methods may be called.

        Raises:
            UndeclaredRead: The instruction does not declare this read.
            EmulatorError: No op is in flight (a handler called outside a
                cycle), or no cycle has taken a snapshot.
        """
        plan = self.plan
        if plan is None:
            raise EmulatorError(
                f"read of {register} with no op issued: a handler ran outside a cycle"
            )
        try:
            from_snapshot = plan.views[_DECLARED_AS.get(register, register), via]
        except KeyError:
            raise self._undeclared(plan, register, via) from None
        return self.snapshot if from_snapshot else self._state.regfile

    def read(self, register: str, via: str | None = None) -> memoryview:
        """Register *register*'s bytes, read-only, in its declared view.

        Args:
            register: See :meth:`source`.
            via: See :meth:`source`.

        Returns:
            A read-only view of the register's bytes; constant-ones provenance
            is kept.
        """
        return self.source(register, via).view(register)

    def read_forgetting_provenance(self, register: str, via: str | None = None) -> memoryview:
        """:meth:`read`, and forget the register's constant-ones provenance in
        the file read. Reading through ``raw()`` used to forget it, and these
        reads still do, so alias and identity statistics stay as they were
        until E04-24."""
        source = self.source(register, via)
        source.forget_provenance(register)
        return source.view(register)

    def lr(self, index: int, via: str | None = None) -> int:
        """LR *index* in the view declared for ``("lr", via)``."""
        return self.source("lr", via).get_lr(index)

    def cr(self, index: int, via: str | None = None) -> int:
        """CR *index* in the view declared for ``("cr", via)``."""
        return self.source("cr", via).get_cr(index)

    def dstructure(self, index: int, via: str | None = None) -> DStructureConfig:
        """CR *index*'s dstructure configuration, read as :meth:`cr` reads it."""
        return decode_dstructure(self.cr(index, via))

    @staticmethod
    def _undeclared(plan: SlotPlan, register: str, via: str | None) -> UndeclaredRead:
        where = f" via {via!r}" if via is not None else ""
        declared = sorted(f"{reg} via {op!r}" if op else reg for reg, op in plan.views)
        return UndeclaredRead(
            f"{plan.inst_name} reads {register}{where}, which it does not declare "
            f"(declared reads: {declared}); add it to the instruction's read "
            "operands or implicit_reads in instruction_spec.py"
        )


def resolver_for(state: IpuState) -> OperandResolver:
    """The operand resolver adapter for *state*: packed when prototyping, else faithful.

    Both return equal operands (:mod:`ipu_emu.resolver`); the packed one reads
    LR and CR words from packed bank views instead of calling an accessor per read.
    """
    return PackedResolver(state) if state.prototyping else FaithfulResolver(state)


class Cycle:
    """Executes the instruction word at the PC, one cycle per call.

    Each PC's :class:`Schedule` is built on first use and cached until that IMEM
    entry is written; :meth:`reset` drops them all.

    Args:
        state: The machine state the cycle reads and writes.
        ipu: The handlers and slot dispatch the cycle drives. Held as given:
            ``Ipu`` passes a weak proxy of itself, so it is not a reference cycle.
        skip_idle_nops: Build schedules that skip idle NOP slots. Defaults to
            ``state.prototyping``: prototyping mode is this flag (and its
            operand resolver) and nothing else. Results are identical either
            way, so a test may pass it in any build.
    """

    def __init__(
        self,
        state: IpuState,
        ipu: Ipu,
        *,
        skip_idle_nops: bool | None = None,
    ) -> None:
        self.state = state
        self._ipu = ipu
        # This cycle's own observers; the state's (IpuState.add_cycle_observer)
        # are added to them at the start of every cycle.
        self._observers: tuple[CycleObserver, ...] = ()
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

    def add_observer(self, observer: CycleObserver) -> Callable[[], None]:
        """Attach *observer* to this cycle; it sees every later cycle this one runs.

        Observers are called in the order they were added, after this cycle's
        state's own (:meth:`~ipu_emu.ipu_state.IpuState.add_cycle_observer`).

        Args:
            observer: The :class:`CycleObserver` to call.

        Returns:
            A function that detaches it; calling it twice is harmless.
        """
        self._observers = self._observers + (observer,)
        ref = weakref.ref(self)

        def remove() -> None:
            cycle = ref()
            if cycle is not None:
                cycle._observers = tuple(o for o in cycle._observers if o is not observer)

        return remove

    @property
    def observers(self) -> tuple[CycleObserver, ...]:
        """Every observer the next cycle calls: the state's, then this cycle's."""
        return self.state.cycle_observers + self._observers

    @property
    def snapshot(self) -> RegFileReader | None:
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
        inst, schedule = self._fetch(self.state.program_counter)
        if schedule is None:
            return CycleOutcome.RETIRED
        self._take_snapshot()
        self._ipu.issuer.observers = self.observers
        return self._evaluate_break(inst, schedule)

    def retire(self) -> None:
        """Take a fresh snapshot, then run every slot except BREAK, in order.

        The snapshot is taken again on purpose: an edit a debugger made while
        stopped at a break is visible to the resumed word, and so is a word it
        wrote into IMEM (that write dropped the cached schedule).
        """
        pc = self.state.program_counter
        inst, schedule = self._fetch(pc)
        if schedule is not None:
            self._take_snapshot()
        self._run(pc, inst, schedule, self.observers)

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
        entry = self._schedules.get(pc)
        inst, schedule = self._fetch(pc) if entry is None else entry
        # Read once per cycle: the state's observers, then this cycle's.
        observers = self.state.cycle_observers + self._observers
        if schedule is not None:
            self._take_snapshot()
            if honour_break and schedule.break_op is not None:
                issuer = self._ipu.issuer
                issuer.observers = observers
                if issuer.dispatch(schedule.break_op, inst) is CycleOutcome.BREAK:
                    return CycleOutcome.BREAK
        self._run(pc, inst, schedule, observers)
        return CycleOutcome.RETIRED

    def reset(self) -> None:
        """Forget cached schedules and the snapshot source.

        The next cycle copies the register file afresh, and no schedule cached
        for an old program survives.
        """
        self._source = None
        self._schedules.clear()

    def _fetch(self, pc: int) -> tuple[Mapping[str, int] | None, Schedule | None]:
        """The word at *pc* and its schedule, from the cache or built now."""
        entry = self._schedules.get(pc)
        if entry is not None:
            return entry
        inst = self.state.inst_mem[pc]
        entry = (
            inst,
            None if inst is None else schedule_word(inst, skip_idle_nops=self._skip_idle_nops),
        )
        # The IMEM write observer reports non-negative PCs, so only those are
        # cached (a negative PC is an error elsewhere, never a hot path).
        if pc >= 0:
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

    def _run(
        self,
        pc: int,
        inst: Mapping[str, int] | None,
        schedule: Schedule | None,
        observers: tuple[CycleObserver, ...],
    ) -> None:
        """Every side effect of one word, between the observers' begin and end."""
        self._ipu.issuer.observers = observers
        for observer in observers:
            observer.begin(pc)
        if inst is None or schedule is None:
            # An idle word: the multiplier sits idle and the PC advances.
            self.state.stats.mult_idle_cycles += 1
            self.state.program_counter += 1
        else:
            self._run_slots(inst, schedule)
        for observer in observers:
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
            raise lr_conflict_error(schedule.lr_conflict)
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
# entering through Ipu's cycle delegates (E08).
from ipu_emu.ipu import plans_for  # noqa: E402
