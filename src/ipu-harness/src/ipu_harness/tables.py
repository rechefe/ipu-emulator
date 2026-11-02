"""A kernel's XMEM tables: declared in ``kernel.toml``, checked against each run.

The compiler that assigns Cache Unit banks needs, for each table a program
uses, how many rows it holds and how the program walks them
(``docs/content/specs/cache-unit.md``). A kernel says so in ``[tables]``
(:mod:`ipu_harness.manifest`): a table is one region reached through its
``[abi]`` base CR, so its rows and its CR come from the manifest and the
placement, and the declaration adds ``repetitions``, ``jump_back``,
``overhead_rows`` and ``whole_run`` as expressions of the case.

:func:`overhead_rows` gives placement the rows to reserve past each region.
:class:`TableRecorder` records a run's XMEM accesses through the reference
model's API, and :func:`check` turns them into a :class:`TableReport`: each
table's :class:`~ipu_harness.table_trace.TableProfile`, and the findings
where the run and the declaration disagree. Errors:

* an access through a CR that no declared table is reached through;
* an access below a table's first row or past its overhead rows;
* a base CR whose value changed while the program used it;
* repetitions other than declared (a backward jump larger than
  ``jump_back`` starts one), or a reach back beyond ``jump_back``;
* an ``also_via`` the program never uses (stale), two tables whose rows
  (overhead included) overlap, more than :data:`MAX_TABLES` tables.

Pending (reported, not failed): a table also reached through another CR
(``also_via``), a base CR holding the base plus an ``[abi]`` offset, a table
read through the hardwired ``CR0``. Each breaks "one table, one CR holding
its table ID", which a later change of the programs fixes. Warnings: declared
overhead the run does not reach, a declared table it never touches.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, XmemAccess
from ipu_harness._plugins import load_file
from ipu_harness.expressions import ExpressionError, evaluate
from ipu_harness.manifest import TABLE_KEYS, KernelManifest, TableDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_harness.placement import Placement
from ipu_harness.table_trace import TableAccess, TableProfile, profile
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES

__all__ = [
    "MAX_TABLES",
    "Finding",
    "Table",
    "TableRecorder",
    "TableReport",
    "check",
    "declares_tables",
    "declared",
    "overhead_rows",
]

#: Tables one program may use: a table ID is 4 bits (the Cache Unit spec).
MAX_TABLES = 16

Severity = Literal["error", "pending", "warning"]


@dataclass(frozen=True)
class Table:
    """One table of a case: its declaration, evaluated, and where it is.

    Attributes:
        name: Its tensor.
        cr: Its base CR's number.
        base_row: Its first XMEM row.
        rows: Its own rows (the region's).
        overhead_rows: Rows reserved past them.
        repetitions: Declared traversals (``None``: not declared).
        jump_back: Declared reach back (``None``: not declared).
        whole_run: Needed from start to end.
        views: Other CRs the program reaches it through, by number, to the
            ``also_via`` name.
        cr_offset: The ``[abi]`` offset its base CR holds past its first row.
    """

    name: str
    cr: int
    base_row: int
    rows: int
    overhead_rows: int = 0
    repetitions: int | None = None
    jump_back: int | None = None
    whole_run: bool = False
    views: Mapping[int, str] = field(default_factory=dict)
    cr_offset: int = 0

    @property
    def declared(self) -> bool:
        """Whether ``[tables]`` declares it (else only measured)."""
        return self.repetitions is not None


@dataclass(frozen=True)
class Finding:
    """Where a run and a declaration disagree.

    Attributes:
        severity: ``error`` fails the case's test; ``pending`` names a rule
            today's program breaks and a later change fixes; ``warning``
            does neither.
        table: The table, or ``CR<n>`` for an access through no table.
        message: What was found.
    """

    severity: Severity
    table: str
    message: str

    def describe(self) -> str:
        """``error: x: 3 accesses past rows=4 + overhead_rows=0 ...``."""
        return f"{self.severity}: {self.table}: {self.message}"


@dataclass(frozen=True)
class TableReport:
    """A run's tables, what it did with each, and the findings.

    Attributes:
        tables: Every table of the case (declared, or every bound region when
            the kernel declares none).
        profiles: The profile of each table the run touched, by name.
        findings: Disagreements with the declarations (none when the kernel
            declares no table).
    """

    tables: tuple[Table, ...]
    profiles: Mapping[str, TableProfile]
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        """The findings that fail the case."""
        return tuple(f for f in self.findings if f.severity == "error")

    def describe(self) -> str:
        """One line per table, then one per finding: the CLI's ``--tables`` report."""
        lines = []
        for table in self.tables:
            seen = self.profiles.get(table.name)
            head = f"{table.name}: CR{table.cr} rows {table.base_row}+{table.rows}"
            if table.overhead_rows:
                head += f" (+{table.overhead_rows} overhead)"
            if seen is None:
                lines.append(f"{head}  never accessed")
                continue
            lines.append(
                f"{head}  reads={seen.reads} writes={seen.writes} rows {seen.lowest}..{seen.highest} "
                f"repetitions={seen.repetitions} jump_back={seen.jump_back}"
            )
        lines.extend(f.describe() for f in self.findings)
        return "\n".join(lines)

    def suggestion(self) -> str:
        """A ``[tables]`` section matching this run, to start a declaration from."""
        lines = ["[tables]"]
        for table in self.tables:
            seen = self.profiles.get(table.name)
            if seen is None:
                continue
            keys = []
            if seen.repetitions != 1:
                keys.append(f"repetitions = {seen.repetitions}")
            if seen.jump_back:
                keys.append(f"jump_back = {seen.jump_back}")
            if seen.overhead_used:
                keys.append(f"overhead_rows = {seen.overhead_used}")
            if table.views:
                keys.append(f"also_via = {sorted(table.views.values())!r}".replace("'", '"'))
            lines.append(
                f"{table.name} = {{ {', '.join(keys)} }}" if keys else f"{table.name} = {{}}"
            )
        return "\n".join(lines)


def declares_tables(manifest: KernelManifest) -> bool:
    """Whether the kernel declares its tables (``[tables]`` or ``[kernel] tables``)."""
    return bool(manifest.tables) or manifest.tables_hook is not None


def _hook_tables(plan: CasePlan, hook: str) -> Mapping[str, TableDecl]:
    """The declarations ``[kernel] tables = "file.py:fn"`` returns for the case."""
    manifest = plan.manifest
    file, _, name = hook.partition(":")
    fn = getattr(load_file(manifest.directory / file), name, None)
    if not callable(fn):
        raise ConfigurationError(f"{manifest.name}: {hook} is no function")
    out: dict[str, TableDecl] = {}
    for tensor, keys in fn(plan).items():
        unknown = sorted(set(keys) - set(TABLE_KEYS))
        if tensor not in manifest.tensors or unknown:
            raise ConfigurationError(
                f"{manifest.name}: {hook} gave table {tensor!r} {dict(keys)}: "
                f"a table is a tensor with keys {list(TABLE_KEYS)}"
            )
        keys = dict(keys)
        keys["also_via"] = tuple(keys.get("also_via", ()))
        out[tensor] = TableDecl(tensor, **keys)
    return out


def _value(plan: CasePlan, decl: TableDecl, key: str) -> int:
    text = getattr(decl, key)
    if isinstance(text, int):
        return text
    ints = {k: v for k, v in plan.params.items() if isinstance(v, int) and not isinstance(v, bool)}
    try:
        value = evaluate(text, {**ints, **plan.derived})
    except ExpressionError as error:
        raise PlanError(plan.manifest, f"tables.{decl.name}.{key}", f"{text!r}: {error}") from None
    if value < 0:
        raise PlanError(plan.manifest, f"tables.{decl.name}.{key}", f"{text!r} is {value} < 0")
    return value


def declared(plan: CasePlan) -> Mapping[str, TableDecl]:
    """The case's table declarations, every expression evaluated to an int.

    Tensors absent from the case (``when``) have none.

    Args:
        plan: The checked case.

    Returns:
        Each declaration, by tensor name, with int values.

    Raises:
        PlanError: An expression does not evaluate, or is negative.
        ConfigurationError: The ``[kernel] tables`` hook is wrong.
    """
    manifest = plan.manifest
    hook = manifest.tables_hook
    decls = manifest.tables if hook is None else _hook_tables(plan, hook)
    return {
        name: TableDecl(
            name,
            repetitions=_value(plan, decl, "repetitions"),
            jump_back=_value(plan, decl, "jump_back"),
            overhead_rows=_value(plan, decl, "overhead_rows"),
            whole_run=decl.whole_run,
            also_via=decl.also_via,
        )
        for name, decl in decls.items()
        if name in plan.shapes
    }


def overhead_rows(plan: CasePlan) -> Mapping[str, int]:
    """The rows placement reserves past each region: its table's overhead rows.

    Args:
        plan: The checked case.

    Returns:
        By tensor name; empty when the kernel declares no table.
    """
    if not declares_tables(plan.manifest):
        return {}
    return {n: int(d.overhead_rows) for n, d in declared(plan).items() if d.overhead_rows}


class TableRecorder:
    """Records every XMEM access of the program a model runs, until stopped.

    Args:
        model: The model, before its run.
    """

    def __init__(self, model: ReferenceModel) -> None:
        self.accesses: list[XmemAccess] = []
        self.stop: Callable[[], None] = model.observe_xmem_accesses(self.accesses.append)


def _cr_number(register: str) -> int | None:
    return int(register[2:]) if register.startswith("CR") and "." not in register else None


def _views(manifest: KernelManifest, name: str, decl: TableDecl | None) -> dict[int, str]:
    """The other CRs a table is reached through, by number: its ``also_via``;
    measured (no declaration), every CR bound to a value of its base alone."""
    if decl is not None:
        names: list[str] = list(decl.also_via)
    else:
        names = [d for d, bases in manifest.placed.items() if bases == {name} and d in manifest.abi]
    views = {}
    for view in names:
        number = 0 if view == "CR0" else _cr_number(manifest.abi[view])
        if number is not None:
            views[number] = view
    return views


def _tables(plan: CasePlan, placement: Placement) -> tuple[Table, ...]:
    """Every region bound to a CR, as a table: declared, or to be measured."""
    manifest = plan.manifest
    decls = declared(plan) if declares_tables(manifest) else {}
    out = []
    for name, region in placement.regions.items():
        cr = _cr_number(manifest.abi.get(name, ""))
        if cr is None or (decls and name not in decls):
            continue
        decl = decls.get(name)
        views = _views(manifest, name, decl)
        out.append(
            Table(
                name=name,
                cr=cr,
                base_row=region.base_row,
                rows=region.rows,
                overhead_rows=region.overhead_rows,
                repetitions=None if decl is None else int(decl.repetitions),
                jump_back=None if decl is None else int(decl.jump_back),
                whole_run=decl is not None and decl.whole_run,
                views=views,
                cr_offset=manifest.abi_offsets.get(name, 0),
            )
        )
    return tuple(out)


def check(
    plan: CasePlan,
    placement: Placement,
    launch: Mapping[int, int],
    accesses: Sequence[XmemAccess],
) -> TableReport:
    """Hold one run's accesses to the case's table declarations.

    A kernel that declares no table gets every region bound to a CR profiled,
    with no finding: a start for its declaration (:meth:`TableReport.suggestion`).

    Args:
        plan: The checked case.
        placement: Where its regions were.
        launch: Each CR's value when the program started, by number (the
            ``[abi]`` writes; the hardwired CR0 and CR1 need not be given).
        accesses: The run's accesses, in program order.

    Returns:
        The report.
    """
    tables = _tables(plan, placement)
    start = {**CR_READ_ONLY_INITIAL_VALUES, **launch}
    by_cr = {t.cr: t for t in tables}
    for table in tables:
        for number in table.views:
            by_cr.setdefault(number, table)
    groups: dict[str, list[TableAccess]] = {t.name: [] for t in tables}
    stray: dict[int, list[XmemAccess]] = {}
    changed: dict[str, set[int]] = {}
    for access in accesses:
        table = by_cr.get(access.cr)
        if table is None:
            stray.setdefault(access.cr, []).append(access)
            continue
        if access.base != _signed(start.get(access.cr, 0)):
            changed.setdefault(table.name, set()).add(access.cr)
        groups[table.name].append(
            TableAccess(
                access.cycle,
                access.pc,
                access.mnemonic,
                access.kind,
                access.cr,
                access.lr,
                access.row - table.base_row,
            )
        )
    profiles = {
        t.name: profile(t.name, t.rows, t.overhead_rows, groups[t.name], _rule(t))
        for t in tables
        if groups[t.name]
    }
    if not declares_tables(plan.manifest):
        measured = [
            Finding("warning", f.table, f.message) for f in _stray_findings(stray, placement)
        ]
        return TableReport(tables, profiles, tuple(measured))
    findings = [
        *_stray_findings(stray, placement),
        *(f for t in tables for f in _table_findings(t, profiles.get(t.name), groups[t.name])),
        *(Finding("error", n, f"CR{sorted(c)} changed while in use") for n, c in changed.items()),
        *_layout_findings(tables, profiles),
    ]
    return TableReport(tables, profiles, tuple(findings))


def _signed(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def _rule(table: Table) -> int | None:
    """``profile``'s jump_back: declared, -1 for a whole-run table, ``None`` to measure."""
    return -1 if table.whole_run else table.jump_back


def _stray_findings(stray: Mapping[int, list[XmemAccess]], placement: Placement) -> list[Finding]:
    out = []
    for cr, seen in sorted(stray.items()):
        rows = sorted({a.row for a in seen})
        hit = sorted(
            {
                r.name
                for r in placement.regions.values()
                for row in rows
                if r.base_row <= row < r.end_row
            }
        )
        out.append(
            Finding(
                "error",
                f"CR{cr}",
                f"{len(seen)} accesses through CR{cr}, which no declared table is reached through "
                f"(rows {rows[0]}..{rows[-1]}, in {hit or 'no region'})",
            )
        )
    return out


def _table_findings(
    table: Table, seen: TableProfile | None, accesses: Sequence[TableAccess]
) -> list[Finding]:
    """One declared table's findings."""
    if seen is None:
        return [Finding("warning", table.name, "declared, never accessed")]
    return [*_pending(table, accesses), *_rows(table, seen), *_walk(table, seen)]


def _pending(table: Table, accesses: Sequence[TableAccess]) -> list[Finding]:
    """The rules today's program breaks, and ``also_via`` entries it never uses."""
    name = table.name
    out = []
    if table.cr == 0:
        out.append(Finding("pending", name, "read through the hardwired CR0 (pinned at row 0)"))
    if table.cr_offset:
        out.append(Finding("pending", name, f"CR{table.cr} holds its base {table.cr_offset:+d}"))
    used = {a.cr for a in accesses}
    for number, view in table.views.items():
        if number in used:
            out.append(Finding("pending", name, f"also reached through CR{number} ({view})"))
        else:
            out.append(Finding("error", name, f"also_via {view!r}: never used (stale)"))
    return out


def _rows(table: Table, seen: TableProfile) -> list[Finding]:
    """Accesses outside the rows and overhead, or overhead the run never reaches."""
    if seen.outside:
        return [Finding("error", table.name, _gap(table, seen))]
    if seen.overhead_used < table.overhead_rows:
        message = f"overhead_rows={table.overhead_rows}; the run reaches {seen.overhead_used}"
        return [Finding("warning", table.name, message)]
    return []


def _walk(table: Table, seen: TableProfile) -> list[Finding]:
    """Repetitions and reach back against the declaration."""
    if table.whole_run or table.repetitions is None or table.jump_back is None:
        return []
    out = []
    # On a one-row table no backward jump can show a new traversal.
    if seen.repetitions != table.repetitions and table.rows + table.overhead_rows > 1:
        extra = sorted(seen.reps[1:], key=lambda r: r.start_jump)[:1]
        hint = f"; the smallest restart jumps back {extra[0].start_jump} rows" if extra else ""
        out.append(
            Finding(
                "error",
                table.name,
                f"{seen.repetitions} repetitions, declared {table.repetitions} "
                f"(a repetition starts at {seen.restart_rule}){hint}",
            )
        )
    worst = max(seen.reps, key=lambda r: r.reach)
    if worst.reach > table.jump_back and worst.reach_from and worst.reach_to:
        out.append(
            Finding(
                "error",
                table.name,
                f"reaches back {worst.reach} rows, declared jump_back={table.jump_back}: "
                f"{worst.reach_from.describe()}, then {worst.reach_to.describe()}",
            )
        )
    return out


def _gap(table: Table, seen: TableProfile, listed: int = 4) -> str:
    """The accesses outside the table's rows, and the overhead that would cover them."""
    outside = seen.outside
    end = table.rows + table.overhead_rows
    parts = [
        f"{len(outside)} accesses outside rows={table.rows} + overhead_rows={table.overhead_rows}"
    ]
    above = [a.row for a in outside if a.row >= end]
    if above:
        parts.append(f"covering them takes overhead_rows={max(above) - table.rows + 1}")
    if any(a.row < 0 for a in outside):
        parts.append("some are below its first row")
    parts.extend(a.describe() for a in outside[:listed])
    return "; ".join(parts)


def _layout_findings(
    tables: Sequence[Table], profiles: Mapping[str, TableProfile]
) -> list[Finding]:
    """More tables than IDs, and tables whose reserved rows overlap."""
    out = []
    used = [t for t in tables if t.name in profiles]
    if len(used) > MAX_TABLES:
        out.append(Finding("error", "*", f"{len(used)} tables; a program has {MAX_TABLES} IDs"))
    spans = sorted(used, key=lambda t: t.base_row)
    for index, table in enumerate(spans):
        end = table.base_row + table.rows + table.overhead_rows
        for other in spans[index + 1 :]:
            if other.base_row >= end:
                break
            out.append(
                Finding("error", table.name, f"its rows and overhead overlap {other.name}'s")
            )
    return out
