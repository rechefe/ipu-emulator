"""Record a kernel's XMEM accesses per table, and derive the Cache Unit's table properties.

Written for kernel programmers who declare their tables for the compiler that
assigns XMEM banks (``docs/content/specs/cache-unit.md``). Each table a kernel
touches is identified by the **CR holding its base**: every XMEM instruction
addresses ``offset (LR) + base (CR)``, and in hardware the table ID travels with
the access the same way. The LR value is therefore the table-relative offset
the Cache Unit sees, so everything here works in offsets, independent of where
the harness happened to place the table.

The model, per table:

* **repetitions** -- traversals of the table's (virtual) data. While
  repetitions remain the table stays resident in XMEM; nothing is freed.
* **jump_back** -- the furthest the cursor reaches back below its high-water
  mark within one repetition, in rows, regardless of where banks fall (a
  margin, not a bank-exact minimum). It matters in the last repetition, when
  banks behind the cursor are released.
* **A new repetition starts at a backward jump larger than jump_back**, a jump
  being the step from one access to the next. With a declared ``jump_back``
  that is the rule exactly; a jump that should not have been a restart shows
  up as an extra repetition, and small steps that add up past jump_back show
  up as reach beyond it. With nothing declared, a jump back across at least
  ``restart_fraction`` (90%) of the table's rows is taken as a restart, to
  suggest starting values.
* **Overhead rows** -- rows past the table's data that the kernel accesses,
  typically a loop's last prefetch. They are real accesses, so they must be
  declared (``overhead_rows``) and allocated: they count for jump_back and for
  banks. An access beyond ``rows + overhead_rows`` (or below the base) fails
  the check, which reports every such access and the gap to declare. Nothing
  here widens a declaration by itself. Today's harnesses place tables back to
  back, so a table's overhead rows are often the next table's first rows; the
  report notes such overlaps rather than failing them, because XMEM placement
  is moving to banks.
* **Whole-run tables** (``whole_run=True``) are needed from start to end of the
  program: never released, all their banks held, no jump_back or repetition
  rule applied.

From that: banks, ring size (banks live at once, ``xmem_num_of_banks``),
physical slots needed, liveness (first/last access -- the last being where an
explicit release goes), and a suggested table type. Across tables
(:func:`liveness`): live banks over time and the order tables start and die in,
in terms of the other tables' progress -- what a bank allocator needs.

Usage::

    from ipu_emu.table_trace import trace_tables

    with trace_tables() as session:
        run_until_complete(state)
    for program in session.programs:
        for profile in program.profiles().values():
            print(profile.cr, profile.jump_back, profile.repetitions)

Every program started inside the block (each ``Ipu`` engine, so each stage of
a multi-stage kernel) is recorded separately. Accesses made outside the engine
-- a harness loading inputs or dumping outputs -- never pass through dispatch
and are not recorded.

This module must not import ``ipu_emu.ipu``: the engine imports it.
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Mapping, NamedTuple

# Spec section 6: a bank is 1024 rows; section 3: 16 banks; section 5:
# ``xmem_num_of_banks`` is 4 bits.
BANK_ROWS = 1024
NUM_BANKS = 16
MAX_TABLE_BANKS = 15

# Without a declared jump_back, a jump back across this fraction of a table's
# rows is taken as the start of a new repetition.
DEFAULT_RESTART_FRACTION = 0.9


@dataclass(frozen=True)
class Table:
    """A kernel's declaration of one table, as a function of its parameters.

    Attributes:
        cr:           The CR whose value is the table's base row.
        jump_back:    Furthest backward reach below the high-water mark within
                      one repetition, in rows, regardless of bank boundaries.
                      A backward jump (from the previous access) larger than
                      this starts a repetition.
        repetitions:  Traversals of the table's data.
        rows:         Rows of data: the size the compiler needs, and what
                      every access is checked against. Required.
        overhead_rows: Rows past ``rows`` the kernel also accesses (a prefetch
                      past the end). Allocated with the table; an access past
                      ``rows + overhead_rows`` is an error. In today's layouts
                      these rows may be the next table's first rows -- noted
                      in the report, not failed.
        whole_run:    The table is needed for the whole program: never
                      released, every bank held. ``jump_back`` and
                      ``repetitions`` are then not used. (Equivalent to a
                      jump_back as large as the table, but says so.)
        program:      Which program run it belongs to, for kernels that run
                      several one after another (the order they start in).

    Several declarations can describe one run correctly (a re-read declared
    as repetitions, or as one repetition with a large jump_back). The check
    verifies that a declaration -- a formula of the parameters -- is
    consistent with every case; it does not pick a unique one.
    """

    cr: int
    jump_back: int = 0
    repetitions: int = 1
    rows: int | None = None
    overhead_rows: int = 0
    whole_run: bool = False
    program: int = 0


class Access(NamedTuple):
    """One XMEM row access, as dispatch saw it."""

    cycle: int
    pc: int
    inst: str
    kind: str       # "read" or "write"
    cr: int         # CR index supplying the base
    base: int       # its value (a row)
    lr: int         # LR index supplying the offset
    offset: int     # row offset within the table (signed)


class Location(NamedTuple):
    """Where in the program an access happened, and at which offset."""

    cycle: int
    pc: int
    inst: str
    offset: int
    lr: int
    cr: int

    @staticmethod
    def of(access: Access) -> "Location":
        return Location(access.cycle, access.pc, access.inst, access.offset,
                        access.lr, access.cr)

    def describe(self) -> str:
        """``offset 12 @ cycle 40, pc 7 (LDR_MULT_REG LR3 + CR2)``."""
        return (f"offset {self.offset} @ cycle {self.cycle}, pc {self.pc} "
                f"({self.inst} LR{self.lr} + CR{self.cr})")


def _signed32(value: int) -> int:
    return value - (1 << 32) if value >= 1 << 31 else value


# -- recording ---------------------------------------------------------------


class ProgramTrace:
    """Every table access of one program run (one ``Ipu`` engine)."""

    def __init__(self, index: int, engine) -> None:
        self.index = index
        self.accesses: list[Access] = []
        self.cycles = 0
        # Kept until the session ends so ``cycles`` can be read then; ``state``
        # stays available so a report can show the instructions it cites.
        self._engine = engine
        self.state = engine.state

    def record(self, cycle, pc, inst, kind, cr, base, lr, offset) -> None:
        """Dispatch callback (see ``Ipu.dispatch_instruction``).

        One access is one row, whatever the element width: tables are
        addressed in rows. (In narrow mode the emulator's 512 B stores also
        overwrite the next three rows; that is the emulator's behaviour, not
        the table model's, and is not recorded.)
        """
        self.accesses.append(Access(cycle, pc, inst, kind, cr, base, lr, _signed32(offset)))

    def finish(self) -> None:
        if self._engine is not None:
            self.cycles = self._engine.cycles_issued
            self._engine = None

    def profiles(self, bank_rows: int = BANK_ROWS,
                 declared: Mapping[int, Table] | None = None,
                 restart_fraction: float = DEFAULT_RESTART_FRACTION,
                 ) -> dict[int, "TableProfile"]:
        """Per-table analysis, keyed by CR index.

        ``declared`` maps a CR to its declared :class:`Table`; those tables are
        analysed under the declared ``jump_back`` and ``rows``, the others by
        the ``restart_fraction`` rule and their measured extent.
        """
        by_cr: dict[int, list[Access]] = {}
        for access in self.accesses:
            by_cr.setdefault(access.cr, []).append(access)
        declared = declared or {}
        return {
            cr: _profile(cr, accesses, bank_rows, self.cycles, declared.get(cr),
                         restart_fraction)
            for cr, accesses in sorted(by_cr.items())
        }


@dataclass
class TraceSession:
    """All programs recorded by one :func:`trace_tables` block, in start order."""

    programs: list[ProgramTrace] = field(default_factory=list)


_ACTIVE: ContextVar[TraceSession | None] = ContextVar("ipu_table_trace", default=None)


def program_observer(engine):
    """Called by ``Ipu.__init__``: a recording callback when tracing is on."""
    session = _ACTIVE.get()
    if session is None:
        return None
    program = ProgramTrace(len(session.programs), engine)
    session.programs.append(program)
    return program.record


@contextmanager
def trace_tables():
    """Record every program started inside the block. Yields the session.

    Results (``ProgramTrace.cycles``, idle tails) are final once it exits.
    """
    session = TraceSession()
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)
        for program in session.programs:
            program.finish()


# -- per-table analysis --------------------------------------------------------


@dataclass
class Repetition:
    """One traversal of a table: from a restart to the next."""

    accesses: list[Access] = field(repr=False)
    # The backward jump that started it (0 for the first), and its two ends.
    start_jump: int = 0
    start_from: Location | None = None
    start_to: Location | None = None
    # Its largest backward reach, and the two accesses that span it.
    reach: int = 0
    reach_from: Location | None = None
    reach_to: Location | None = None

    @property
    def first(self) -> Location:
        return Location.of(self.accesses[0])

    @property
    def last(self) -> Location:
        return Location.of(self.accesses[-1])


def _split(accesses: list[Access], is_restart: Callable[[int], bool]) -> list[Repetition]:
    """Split accesses into repetitions, measuring each one's reach.

    A backward jump from the previous access for which ``is_restart(distance)``
    holds starts a new repetition. Reach is measured from the repetition's
    high-water mark, so a slow walk backwards accumulates reach without ever
    being a restart.
    """
    reps: list[Repetition] = []
    current: Repetition | None = None
    high: Access | None = None
    previous: Access | None = None
    for access in accesses:
        if previous is not None:
            back = previous.offset - access.offset
            if back > 0 and is_restart(back):
                current = Repetition([], back, Location.of(previous), Location.of(access))
                reps.append(current)
                high = access
        previous = access
        if current is None:
            current = Repetition([])
            reps.append(current)
            high = access
        current.accesses.append(access)
        if access.offset > high.offset:
            high = access
        elif high.offset - access.offset > current.reach:
            current.reach = high.offset - access.offset
            current.reach_from, current.reach_to = Location.of(high), Location.of(access)
    return reps


def _simulate_banks(reps: list[Repetition], inside: Callable[[int], bool],
                    bank_rows: int, jump_back: int):
    """Banks live over time, from in-table accesses only.

    Earlier repetitions keep the table resident; only in the last one is a
    bank released, once an access in the next bank reaches in-bank offset
    ``jump_back`` (spec sections 9.1/9.2). Banks never released stay live until
    the table's last access, where the explicit release goes.

    Returns (most banks live at once, physical slots needed, intervals). Banks
    are counted before an access releases anything: the bank it lands in must
    be present while the one below is still held, so a forward stream needs
    two. ``slots`` is the smallest ring whose ``bank % slots`` mapping never
    puts two live banks in one slot (spec section 9.1's ``bank_pos``).
    """
    live: dict[int, int] = {}            # bank -> first cycle
    intervals: list[tuple[int, int, int]] = []
    snapshots: set[frozenset[int]] = set()
    ring = 0
    end = None
    last = len(reps) - 1
    for index, rep in enumerate(reps):
        for access in rep.accesses:
            if not inside(access.offset):
                continue
            end = access.cycle
            bank = access.offset // bank_rows
            if bank not in live:
                live[bank] = access.cycle
                snapshots.add(frozenset(live))
                ring = max(ring, len(live))
            if (index == last and bank >= 1 and access.offset % bank_rows >= jump_back
                    and bank - 1 in live):
                intervals.append((bank - 1, live.pop(bank - 1), access.cycle))
    if end is not None:
        intervals.extend((bank, start, end) for bank, start in live.items())
    slots = ring
    while any(len({b % slots for b in snap}) < len(snap) for snap in snapshots):
        slots += 1
    return ring, slots, sorted(intervals)


def _suggest_type(accesses: list[Access], rows: int) -> str:
    """From every allocated access; a prefetch read of an overhead row that
    was never written does not make a staging table need DRAM contents."""
    kinds = {a.kind for a in accesses}
    if kinds == {"read"}:
        return "read_only"
    if kinds == {"write"}:
        return "write_only"
    written: set[int] = set()
    for access in (a for a in accesses if a.offset < rows):
        if access.kind == "write":
            written.add(access.offset)
        elif access.offset not in written:
            # Reads data it did not write: needs DRAM contents *and* writes.
            return "read+write"
    return "read_after_write"


def _write_order(accesses: list[Access]) -> tuple[int, int]:
    seen: set[int] = set()
    rewritten: set[int] = set()
    high = None
    out_of_order = 0
    for access in accesses:
        if access.kind != "write":
            continue
        if access.offset in seen:
            rewritten.add(access.offset)
        seen.add(access.offset)
        if high is not None and access.offset < high:
            out_of_order += 1
        high = access.offset if high is None else max(high, access.offset)
    return len(rewritten), out_of_order


@dataclass
class TableProfile:
    """Everything measured about one table (one base CR) in one program."""

    cr: int
    declared: Table | None
    bases: tuple[int, ...]          # distinct base values seen; >1 is a bug
    reads: int
    writes: int
    min_offset: int
    max_offset: int
    rows: int                       # declared, else highest offset + 1
    overhead_rows: int              # declared, else 0
    overreads: list[Location]       # accesses outside [0, rows): overhead or worse
    reps: list[Repetition]
    restart_rule: str               # how repetitions were split, for display
    bank_rows: int
    ring_banks: int                 # most banks live at once
    slots: int                      # ring size free of slot collisions
    bank_live: list[tuple[int, int, int]]  # (bank, first cycle, release cycle)
    first: Location
    last: Location
    program_cycles: int
    suggested_type: str
    rows_rewritten: int             # rows written more than once
    out_of_order_writes: int        # writes below the write high-water mark

    @property
    def jump_back(self) -> int:
        """Largest backward reach within any repetition."""
        return max(r.reach for r in self.reps)

    @property
    def repetitions(self) -> int:
        return len(self.reps)

    @property
    def worst(self) -> Repetition:
        """The repetition with the largest reach."""
        return max(self.reps, key=lambda r: r.reach)

    @property
    def allocated_rows(self) -> int:
        """Rows the table is allocated: data plus declared overhead."""
        return self.rows + self.overhead_rows

    @property
    def undeclared(self) -> list[Location]:
        """Accesses outside the allocated rows: below the base or past the overhead."""
        return [o for o in self.overreads if not 0 <= o.offset < self.allocated_rows]

    @property
    def overhead_used(self) -> int:
        """Rows past ``rows`` the kernel actually reaches."""
        return max((o.offset - self.rows + 1 for o in self.overreads
                    if o.offset >= self.rows), default=0)

    @property
    def banks(self) -> int:
        """Banks the table's allocated rows occupy."""
        return max(1, -(-self.allocated_rows // self.bank_rows))

    @property
    def resident(self) -> bool:
        """Every bank stays live at once (repeated, or never released)."""
        return self.ring_banks >= self.banks

    @property
    def idle_tail(self) -> int:
        """Cycles after the last access until the program ends."""
        return self.program_cycles - self.last.cycle - 1

    @property
    def flat_base(self) -> bool:
        """Addressed through CR0 (hardwired 0) with absolute rows in the LR.

        Attributable only while a single table, sitting at row 0, is reached
        that way; any other table behind CR0 is merged into it. The fix is a
        base CR of its own for each table."""
        return self.cr == 0

    def suggestion(self, program: int = 0) -> Table:
        """A declaration matching this run, to start a formula from."""
        return Table(cr=self.cr, jump_back=self.jump_back,
                     repetitions=self.repetitions, rows=self.rows, program=program)


def _profile(cr: int, accesses: list[Access], bank_rows: int, program_cycles: int,
             declared: Table | None, restart_fraction: float) -> TableProfile:
    offsets = [a.offset for a in accesses]
    lo, hi = min(offsets), max(offsets)
    if declared is not None and declared.rows is not None:
        rows = declared.rows
    else:
        rows = hi + 1
    overhead = declared.overhead_rows if declared is not None else 0

    def inside(offset: int) -> bool:
        """Allocated: data rows plus declared overhead."""
        return 0 <= offset < rows + overhead

    # Accesses outside the allocation are already an error of their own;
    # leaving them out keeps them from also splitting repetitions or reach.
    allocated = [a for a in accesses if inside(a.offset)] or accesses
    if declared is not None and declared.whole_run:
        jump_back = rows + overhead
        reps = _split(allocated, lambda back: False)
        rule = "whole run"
    elif declared is not None:
        jump_back = declared.jump_back
        reps = _split(allocated, lambda back: back > jump_back)
        rule = f"backward jump > jump_back={jump_back}"
    else:
        # The largest possible jump is rows - 1, so the fraction is of that.
        threshold = max(1.0, restart_fraction * (rows - 1))
        reps = _split(allocated, lambda back: back >= threshold)
        jump_back = max(r.reach for r in reps)
        rule = f"backward jump >= {restart_fraction:.0%} of {rows} rows"
    ring, slots, intervals = _simulate_banks(reps, inside, bank_rows, jump_back)
    rewritten, out_of_order = _write_order([a for a in allocated if a.offset < rows])
    return TableProfile(
        cr=cr,
        declared=declared,
        bases=tuple(dict.fromkeys(a.base for a in accesses)),
        reads=sum(a.kind == "read" for a in accesses),
        writes=sum(a.kind == "write" for a in accesses),
        min_offset=lo,
        max_offset=hi,
        rows=rows,
        overhead_rows=overhead,
        overreads=[Location.of(a) for a in accesses if not 0 <= a.offset < rows],
        reps=reps,
        restart_rule=rule,
        bank_rows=bank_rows,
        ring_banks=ring,
        slots=slots,
        bank_live=intervals,
        first=Location.of(accesses[0]),
        last=Location.of(accesses[-1]),
        program_cycles=program_cycles,
        suggested_type=_suggest_type(allocated, rows),
        rows_rewritten=rewritten,
        out_of_order_writes=out_of_order,
    )


# -- across tables -------------------------------------------------------------


class Progress(NamedTuple):
    """How far a table had got at some moment: its repetition and high-water."""

    cr: int
    repetition: int     # 0-based
    offset: int


class Event(NamedTuple):
    """A table starting or ending its use, with the others' progress then."""

    cycle: int
    kind: str           # "first" or "last"
    cr: int
    others: tuple[Progress, ...]


@dataclass
class Liveness:
    """Bank demand over one program's run, and the order tables start and die."""

    peak_banks: int
    peak_cycle: int
    events: list[Event]


def _progress_at(profile: TableProfile, cycle: int) -> Progress | None:
    """A table's repetition and high-water offset as of ``cycle`` (inclusive)."""
    starts = [rep.accesses[0].cycle for rep in profile.reps]
    index = bisect.bisect_right(starts, cycle) - 1
    if index < 0:
        return None
    seen = [a.offset for a in profile.reps[index].accesses if a.cycle <= cycle]
    return Progress(profile.cr, index, max(seen))


def liveness(profiles: Mapping[int, TableProfile]) -> Liveness:
    """Combine per-table bank intervals into the program's bank demand."""
    deltas: list[tuple[int, int]] = []
    for profile in profiles.values():
        for _, start, end in profile.bank_live:
            deltas.append((start, 1))
            deltas.append((end + 1, -1))
    live = peak = peak_cycle = 0
    for cycle, delta in sorted(deltas):
        live += delta
        if live > peak:
            peak, peak_cycle = live, cycle

    def others_at(cr: int, cycle: int) -> tuple[Progress, ...]:
        out = []
        for other, profile in profiles.items():
            if other != cr and profile.first.cycle <= cycle <= profile.last.cycle:
                progress = _progress_at(profile, cycle)
                if progress is not None:
                    out.append(progress)
        return tuple(out)

    events = []
    for cr, profile in profiles.items():
        events.append(Event(profile.first.cycle, "first", cr, others_at(cr, profile.first.cycle)))
        events.append(Event(profile.last.cycle, "last", cr, others_at(cr, profile.last.cycle)))
    # Within a cycle, starts before ends: a table ending in the cycle another
    # starts still holds its banks, so the newcomer cannot take them yet.
    events.sort(key=lambda e: (e.cycle, e.kind == "last"))
    return Liveness(peak, peak_cycle, events)


# -- checking a declaration ------------------------------------------------------


class Finding(NamedTuple):
    severity: str   # "error" or "warning"
    table: str      # declared name, or "cr<N>" for an undeclared one
    message: str


def check_tables(session: TraceSession, tables: Mapping[str, Table],
                 bank_rows: int = BANK_ROWS) -> list[Finding]:
    """Compare a kernel's declared tables with what a run actually did.

    Errors: an access through an undeclared CR, two tables on one CR, a base
    that changes in use, a repetition count that differs (a backward jump
    larger than jump_back starts one), or reach within a repetition beyond
    jump_back, or an access outside ``rows + overhead_rows`` (every such
    access is listed, with the gap to declare). Warnings: declared overhead
    the run does not use, tables never accessed.
    None of this depends on bank size; ``bank_rows`` only feeds the profiles.
    """
    findings: list[Finding] = []
    by_program: dict[int, dict[int, str]] = {}
    for name, table in tables.items():
        crs = by_program.setdefault(table.program, {})
        if table.cr in crs:
            findings.append(Finding("error", name,
                                    f"shares CR{table.cr} with {crs[table.cr]!r}"))
            continue
        crs[table.cr] = name

    for program in session.programs:
        declared = by_program.get(program.index, {})
        profiles = program.profiles(bank_rows, {cr: tables[n] for cr, n in declared.items()})
        for cr, profile in profiles.items():
            name = declared.get(cr)
            if name is None:
                hint = (" -- CR0 is a flat base (absolute rows in the LR); give "
                        "each table a base CR of its own" if profile.flat_base else "")
                findings.append(Finding(
                    "error", f"cr{cr}",
                    f"program {program.index}: {profile.reads + profile.writes} "
                    f"accesses through undeclared CR{cr}{hint}"))
                continue
            findings.extend(_check_one(name, tables[name], profile))
        findings.extend(_check_banks(program, profiles, declared))
        for cr, name in declared.items():
            if cr not in profiles:
                findings.append(Finding("warning", name,
                                        f"program {program.index}: never accessed"))
    for program_index in by_program.keys() - {p.index for p in session.programs}:
        findings.append(Finding("error", "*", f"tables declared for program "
                                f"{program_index}, which never ran"))
    return findings


def _check_banks(program: ProgramTrace, profiles: Mapping[int, TableProfile],
                 names: Mapping[int, str]) -> list[Finding]:
    """Bank demand beyond what the Cache Unit has: warnings to raise, not failures."""
    out = []
    for cr, profile in profiles.items():
        if profile.ring_banks > MAX_TABLE_BANKS:
            out.append(Finding("warning", names.get(cr, f"cr{cr}"),
                               f"needs {profile.ring_banks} banks at once; a table's ring "
                               f"holds at most {MAX_TABLE_BANKS} (xmem_num_of_banks)"))
    peak = liveness(profiles).peak_banks
    if peak > NUM_BANKS:
        out.append(Finding("warning", "*",
                           f"program {program.index} needs {peak} banks at once; XMEM has "
                           f"{NUM_BANKS} -- raise this"))
    return out


def _check_one(name: str, table: Table, profile: TableProfile) -> list[Finding]:
    out: list[Finding] = []
    if table.rows is None:
        out.append(Finding("error", name, "declares no rows: without them no access can "
                           f"be checked against the table (the run reaches offset "
                           f"{profile.max_offset})"))
        return out
    if len(profile.bases) > 1:
        out.append(Finding("error", name, f"CR{table.cr} changed while in use: "
                           f"bases {list(profile.bases)}"))
    if profile.undeclared:
        out.append(Finding("error", name, _gap_message(table, profile)))
    elif profile.overhead_used < table.overhead_rows:
        out.append(Finding("warning", name,
                           f"declares overhead_rows={table.overhead_rows}, the run "
                           f"reaches {profile.overhead_used} row(s) past rows={profile.rows}"))
    if table.whole_run:
        return out
    found, wanted = profile.repetitions, table.repetitions
    if found != wanted:
        message = f"{found} repetitions, declared {wanted}"
        if found > wanted:
            # The smallest restart jumps are the least likely to be intended.
            extra = sorted(profile.reps[1:], key=lambda r: r.start_jump)[0]
            message += (f"; e.g. a jump back of {extra.start_jump} rows (> jump_back="
                        f"{table.jump_back}) from {extra.start_from.describe()} "
                        f"to {extra.start_to.describe()}")
        else:
            message += (f"; a repetition starts at a backward jump larger than "
                        f"jump_back={table.jump_back}")
        out.append(Finding("error", name, message))
    if profile.jump_back > table.jump_back:
        worst = profile.worst
        out.append(Finding(
            "error", name,
            f"reaches back {profile.jump_back} rows, declared jump_back={table.jump_back}: "
            f"{worst.reach_from.describe()}, then {worst.reach_to.describe()}"))
    return out


def _gap_message(table: Table, profile: TableProfile, listed: int = 6) -> str:
    """Every access outside the allocated rows, and the gap to the declaration."""
    rows, allocated = profile.rows, profile.allocated_rows
    outside = profile.undeclared
    parts = [f"{len(outside)} accesses outside its declared rows={rows} + "
             f"overhead_rows={table.overhead_rows} (offsets 0..{allocated - 1})"]
    above = [o.offset for o in outside if o.offset >= allocated]
    below = [o.offset for o in outside if o.offset < 0]
    if above:
        needed = max(above) - rows + 1
        parts.append(f"GAP: offsets {allocated}..{max(above)} are accessed but not "
                     f"declared ({max(above) - allocated + 1} row(s)); covering them "
                     f"takes overhead_rows={needed} -- if the kernel relies on them "
                     f"(a prefetch), declare it; otherwise review or debug the kernel")
    if below:
        parts.append(f"offsets {min(below)}..{max(below)} are below the table's base")
    by_offset: dict[int, list[Location]] = {}
    for location in outside:
        by_offset.setdefault(location.offset, []).append(location)
    lines = [f"offset {offset}: {len(locs)}x, first {locs[0].describe()}"
             for offset, locs in sorted(by_offset.items())[:listed]]
    if len(by_offset) > listed:
        lines.append(f"... and {len(by_offset) - listed} more offsets")
    return "; ".join(parts) + "\n      " + "\n      ".join(lines)
