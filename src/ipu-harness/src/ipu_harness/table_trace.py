"""What a program did with each XMEM table: repetitions, reach back, overhead.

The analysis behind :mod:`ipu_harness.tables`, on the accesses one run made
(:meth:`ipu_emu.reference.ReferenceModel.observe_xmem_accesses`), each turned
into a row of its table (:class:`TableAccess`). It knows nothing of
manifests: the caller says which accesses belong to which table.

The model, per table (``docs/content/specs/cache-unit.md``):

* **repetitions**: traversals of the table's rows. While one remains the
  table stays resident; nothing is released.
* **jump_back**: the furthest the program reaches back below the highest row
  it has reached within one repetition, in rows, wherever banks fall.
* **A repetition starts at a backward jump larger than jump_back**, a jump
  being the step from one access of the table to its next. Small steps back
  that add up past ``jump_back`` show as reach beyond it, not as a restart.
  With no declaration, a jump back across at least :data:`RESTART_FRACTION`
  of the rows is taken as a restart, to suggest starting values.
* **Overhead rows**: rows past the table's own that the program reaches (a
  loop's last prefetch). They are real accesses: declared, reserved, and
  counted for reach. An access below the table's first row or past its
  overhead is an error.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "RESTART_FRACTION",
    "Repetition",
    "TableAccess",
    "TableProfile",
    "profile",
    "split",
]

#: With no declared ``jump_back``, a jump back across this fraction of the
#: table's rows starts a repetition.
RESTART_FRACTION = 0.9


@dataclass(frozen=True)
class TableAccess:
    """One access to a table: where it was in the program and which row it hit.

    Attributes:
        cycle: The words retired before it.
        pc: The word's program counter.
        mnemonic: The instruction.
        kind: ``read`` or ``write``.
        cr: The base CR's number.
        lr: The offset LR's number.
        row: The row within the table: 0 is its first.
    """

    cycle: int
    pc: int
    mnemonic: str
    kind: str
    cr: int
    lr: int
    row: int

    def describe(self) -> str:
        """``row 12 at cycle 40, PC 7 (LD.MULT LR3 CR2)``."""
        return (
            f"row {self.row} at cycle {self.cycle}, PC {self.pc} "
            f"({self.mnemonic} LR{self.lr} CR{self.cr})"
        )


@dataclass
class Repetition:
    """One traversal of a table, from the jump that started it to the next.

    Attributes:
        accesses: Its accesses, in order.
        start_jump: The backward jump that started it (0 for the first).
        reach: Its largest reach back below its highest row so far.
        reach_from: The access at that highest row.
        reach_to: The access that reached furthest below it.
    """

    accesses: list[TableAccess] = field(default_factory=list)
    start_jump: int = 0
    reach: int = 0
    reach_from: TableAccess | None = None
    reach_to: TableAccess | None = None


def split(accesses: Sequence[TableAccess], restarts: Callable[[int], bool]) -> list[Repetition]:
    """Split a table's accesses into repetitions, measuring each one's reach.

    Args:
        accesses: The table's accesses, in program order.
        restarts: Whether a backward jump of this many rows starts a repetition.

    Returns:
        The repetitions (none for no access).
    """
    if not accesses:
        return []
    reps = [Repetition()]
    previous = high = accesses[0]
    for access in accesses:
        back = previous.row - access.row
        if back > 0 and restarts(back):
            reps.append(Repetition(start_jump=back))
            high = access
        previous = access
        current = reps[-1]
        current.accesses.append(access)
        if access.row > high.row:
            high = access
        elif high.row - access.row > current.reach:
            current.reach = high.row - access.row
            current.reach_from, current.reach_to = high, access
    return reps


@dataclass(frozen=True)
class TableProfile:
    """What one run did with one table.

    Attributes:
        name: The table (its tensor's name).
        rows: Its own rows.
        overhead_rows: The overhead rows it was given.
        accesses: Every access, in program order.
        reps: The repetitions of its in-range accesses.
        restart_rule: How they were split, for a report.
    """

    name: str
    rows: int
    overhead_rows: int
    accesses: tuple[TableAccess, ...]
    reps: tuple[Repetition, ...]
    restart_rule: str

    @property
    def reads(self) -> int:
        """Rows read."""
        return sum(a.kind == "read" for a in self.accesses)

    @property
    def writes(self) -> int:
        """Rows written."""
        return sum(a.kind == "write" for a in self.accesses)

    @property
    def repetitions(self) -> int:
        """Repetitions found."""
        return len(self.reps)

    @property
    def jump_back(self) -> int:
        """The largest reach back within any repetition."""
        return max((r.reach for r in self.reps), default=0)

    @property
    def outside(self) -> tuple[TableAccess, ...]:
        """Accesses below the first row or past the overhead rows."""
        end = self.rows + self.overhead_rows
        return tuple(a for a in self.accesses if not 0 <= a.row < end)

    @property
    def overhead_used(self) -> int:
        """Rows past its own that the run reaches."""
        return max((a.row - self.rows + 1 for a in self.accesses if a.row >= self.rows), default=0)

    @property
    def lowest(self) -> int:
        """The lowest row accessed."""
        return min(a.row for a in self.accesses)

    @property
    def highest(self) -> int:
        """The highest row accessed."""
        return max(a.row for a in self.accesses)


def profile(
    name: str,
    rows: int,
    overhead_rows: int,
    accesses: Sequence[TableAccess],
    jump_back: int | None,
) -> TableProfile:
    """Analyse one table's accesses.

    Args:
        name: The table.
        rows: Its own rows.
        overhead_rows: Its overhead rows.
        accesses: Its accesses, in program order (at least one).
        jump_back: The declared jump back, or ``None``: none declared (the
            :data:`RESTART_FRACTION` rule splits repetitions), or a whole-run
            table (``-1``: never split).

    Returns:
        The profile. Accesses outside the table's rows and overhead are in
        it, but split no repetition and add no reach.
    """
    end = rows + overhead_rows
    inside = [a for a in accesses if 0 <= a.row < end] or list(accesses)
    if jump_back is None:
        threshold = max(1.0, RESTART_FRACTION * (end - 1))
        reps = split(inside, lambda back: back >= threshold)
        rule = f"a backward jump of at least {RESTART_FRACTION:.0%} of {end} rows"
    elif jump_back < 0:
        reps = split(inside, lambda back: False)
        rule = "whole run: none"
    else:
        reps = split(inside, lambda back: back > jump_back)
        rule = f"a backward jump larger than jump_back={jump_back}"
    return TableProfile(name, rows, overhead_rows, tuple(accesses), tuple(reps), rule)
