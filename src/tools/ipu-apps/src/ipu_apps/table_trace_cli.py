"""Measure a kernel's XMEM tables, to help write and check its ``SPEC.tables``.

Every table is identified by the CR holding its base (``offset(LR) + base(CR)``
addressing), exactly as the Cache Unit identifies it by table ID. For each one
this reports what the compiler needs to assign banks -- size, ``jump_back``,
``repetitions``, banks live at once, and when the table is live -- together
with the instructions responsible, so a surprising number can be traced to the
loop that causes it. See ``ipu_emu.table_trace`` for the model.

Usage::

    bazel run //src/tools/ipu-apps:table_trace -- --kernel matmul_128x128
    bazel run //src/tools/ipu-apps:table_trace -- --kernel layernorm_64x192 --bank-rows 16
    bazel run //src/tools/ipu-apps:table_trace -- --kernel matmul_128x128 --all-cases
    bazel run //src/tools/ipu-apps:table_trace -- --all-family softmax

A kernel's own runner prints the same report for any case and case options::

    bazel run //src/tools/ipu-apps:<kernel> -- --case default --tables

``--bank-rows`` (default 1024, the real bank) sets the bank size for the
bank-level numbers (banks, ring, slots, liveness). Test cases are small, so
most tables fit in one real bank; a small value such as 16 shows how the
kernel behaves once a table spans banks. jump_back and repetitions do not
depend on it.

Instructions are shown as the debugger's pipeline pane shows them, with the
same PC and cycle numbers, so ``break``/``until`` in the debugger reach them.

Without Bazel (see MEMORY.md)::

    PYTHONPATH=... python -m ipu_apps.table_trace_cli --kernel matmul_128x128
"""

from __future__ import annotations

import argparse
import sys

from ipu_emu.table_trace import (
    BANK_ROWS, DEFAULT_RESTART_FRACTION, MAX_TABLE_BANKS, NUM_BANKS, Location,
    ProgramTrace, TableProfile, liveness,
)

from ipu_apps.kernel_registry.cases import load_cases
from ipu_apps.kernel_registry.registry import kernels
from ipu_apps.kernel_registry.tables import (
    TracedRun, TracedRunFailed, check_declared, run_traced,
)


def _instruction(program: ProgramTrace, location: Location) -> str:
    """The slot that made the access, rendered as the debugger's pipeline pane does."""
    from ipu_emu.debug_pipeline import display_operation, stage_operations

    for _, operation in stage_operations(program.state.inst_mem[location.pc]):
        if operation.split()[0].upper() == location.inst.upper():
            return display_operation(operation)
    return f"{location.inst} LR{location.lr}, CR{location.cr}"


def _where(program: ProgramTrace, location: Location | None) -> str:
    if location is None:
        return "-"
    return (f"offset {location.offset} @ cycle {location.cycle}, pc {location.pc}: "
            f"{_instruction(program, location)}")


def _names(run: TracedRun, program: ProgramTrace) -> dict[int, str]:
    declared = run.declared or {}
    return {t.cr: name for name, t in declared.items() if t.program == program.index}


def _label(names: dict[int, str], cr: int) -> str:
    return f"{names[cr]} (CR{cr})" if cr in names else f"CR{cr}"


def _declared(run: TracedRun, program: ProgramTrace):
    return {t.cr: t for t in (run.declared or {}).values() if t.program == program.index}


def _report_program(run: TracedRun, program: ProgramTrace, bank_rows: int,
                    restart_fraction: float) -> None:
    names = _names(run, program)
    profiles = program.profiles(bank_rows, _declared(run, program), restart_fraction)
    if len(run.session.programs) > 1:
        print(f"\n  program {program.index} ({program.cycles} cycles)")
    if not profiles:
        print("    no XMEM accesses")
        return

    print(f"    {'table':<20} {'type':<17} {'rows':>6} {'jump_back':>9} {'reps':>5} "
          f"{'banks':>5} {'ring':>4} {'slots':>5} {'live (cycles)':>17} {'idle':>7}  r/w")
    for cr, p in profiles.items():
        live = f"{p.first.cycle}..{p.last.cycle}"
        rows = _rows_text(p)
        print(f"    {_label(names, cr):<20} {p.suggested_type:<17} {rows:>6} "
              f"{p.jump_back:>9} {p.repetitions:>5} {p.banks:>5} {p.ring_banks:>4} "
              f"{p.slots:>5} {live:>17} {p.idle_tail:>7}  {p.reads}/{p.writes}")

    notes = _overlaps(profiles, names)
    for cr, p in profiles.items():
        _explain_table(program, _label(names, cr), p, notes.get(cr, []))
    _report_liveness(profiles, names)


def _overlaps(profiles, names) -> dict[int, list[str]]:
    """Declared tables whose allocated rows overlap another's: where the
    "one table, one ID" error (see ``check_tables``) is, table by table."""
    spans = {cr: (p.bases[0], p.bases[0] + p.allocated_rows)
             for cr, p in profiles.items() if p.declared is not None}
    notes: dict[int, list[str]] = {}
    for cr, (start, end) in spans.items():
        for other, (o_start, o_end) in spans.items():
            if other != cr and start < o_end and o_start < end:
                lo, hi = max(start, o_start), min(end, o_end) - 1
                notes.setdefault(cr, []).append(
                    f"!! its allocated rows overlap {_label(names, other)} at XMEM rows "
                    f"{lo}..{hi}: one table per ID, overhead rows included")
    return notes


def _explain_table(program: ProgramTrace, label: str, p: TableProfile,
                   notes: list[str]) -> None:
    lines = []
    if p.flat_base:
        lines.append("addressed through CR0 (flat base, absolute rows in the LR): only valid "
                     "while it is the one table at row 0 -- give it a base CR of its own")
    if len(p.bases) > 1:
        lines.append(f"base CR changed while in use: {list(p.bases)}")
    if p.declared is not None and p.undeclared:
        lines.extend(_gap_lines(program, p))
    elif p.overreads:
        lines.append(f"{len(p.overreads)} accesses in its {p.overhead_rows} overhead row(s) "
                     f"past rows={p.rows}, first {_where(program, p.overreads[0])}")
    if p.jump_back:
        worst = p.worst
        lines.append(f"jump_back {p.jump_back}: from {_where(program, worst.reach_from)}")
        lines.append(f"{'':>{len(str(p.jump_back)) + 11}}back to {_where(program, worst.reach_to)}")
    if p.declared is not None and p.declared.whole_run:
        lines.append("needed for the whole run (declared): never released")
    elif p.repetitions > 1:
        lines.append(f"{p.repetitions} repetitions ({p.restart_rule}); resident until the last")
    if p.ring_banks > MAX_TABLE_BANKS:
        lines.append(f"needs {p.ring_banks} banks live at once: more than xmem_num_of_banks "
                     f"can hold ({MAX_TABLE_BANKS})")
    elif p.slots > p.ring_banks:
        lines.append(f"{p.ring_banks} banks live at once, but bank % N mapping needs "
                     f"{p.slots} slots to avoid collisions")
    # Write order only matters for a table drained to DRAM as it is written;
    # a staging buffer that is written and read back may be reused freely.
    if p.suggested_type == "write_only":
        if p.out_of_order_writes:
            lines.append(f"{p.out_of_order_writes} writes land below earlier writes: "
                         f"cannot be written out in order")
        if p.rows_rewritten:
            lines.append(f"{p.rows_rewritten} rows written more than once: a bank "
                         f"drained to DRAM on its last row would lose the rewrite")
    lines.append(f"last access (release point): {_where(program, p.last)}")
    lines.extend(notes)
    if p.declared is not None:
        d = p.declared
        shape = ("whole_run" if d.whole_run
                 else f"jump_back={d.jump_back} repetitions={d.repetitions}")
        lines.append(f"declared: {shape}"
                     + (f" rows={d.rows}" if d.rows is not None else " rows=MISSING")
                     + (f" overhead_rows={d.overhead_rows}" if d.overhead_rows else ""))
    else:
        s = p.suggestion(program.index)
        lines.append(f"suggest:  Table(cr={s.cr}, jump_back={s.jump_back}, "
                     f"repetitions={s.repetitions}, rows={s.rows}"
                     + (f", program={s.program}" if s.program else "") + ")")
        lines.append(f"          rows={s.rows} is the highest offset + 1: if the last rows are "
                     f"a prefetch past the data, declare them as overhead_rows instead")
    print(f"\n    {label}")
    for line in lines:
        print(f"      {line}")


def _rows_text(p: TableProfile) -> str:
    """``192``; ``192+1`` with declared overhead; ``!`` when accesses fall
    outside what is declared (the gap itself is listed under the table)."""
    if p.declared is not None and p.undeclared:
        return f"{p.rows}+{p.overhead_rows}!"
    return f"{p.rows}+{p.overhead_rows}" if p.overhead_rows else str(p.rows)


def _gap_lines(program: ProgramTrace, p: TableProfile, listed: int = 6) -> list[str]:
    """The undeclared accesses of a declared table, the gap first."""
    allocated = p.allocated_rows
    above = sorted({o.offset for o in p.undeclared if o.offset >= allocated})
    below = sorted({o.offset for o in p.undeclared if o.offset < 0})
    lines = [f"!! {len(p.undeclared)} accesses outside the declaration "
             f"(rows={p.rows} + overhead_rows={p.overhead_rows}: offsets 0..{allocated - 1})"]
    if above:
        lines.append(f"!! GAP: offsets {allocated}..{above[-1]} accessed but not declared; "
                     f"overhead_rows={above[-1] - p.rows + 1} would cover them -- declare it "
                     f"if the kernel relies on them (a prefetch), else review or debug")
    if below:
        lines.append(f"!! offsets {below[0]}..{below[-1]} are below the table's base")
    by_offset: dict[int, list[Location]] = {}
    for location in p.undeclared:
        by_offset.setdefault(location.offset, []).append(location)
    for offset, locations in sorted(by_offset.items())[:listed]:
        lines.append(f"   offset {offset}: {len(locations)}x, first "
                     f"{_where(program, locations[0])}")
    if len(by_offset) > listed:
        lines.append(f"   ... and {len(by_offset) - listed} more offsets")
    return lines


def _report_liveness(profiles, names) -> None:
    live = liveness(profiles)
    warn = f"  -- more than the {NUM_BANKS} XMEM banks" if live.peak_banks > NUM_BANKS else ""
    print(f"\n    banks live at once: peak {live.peak_banks} at cycle {live.peak_cycle}{warn}")
    print("    first and last use of each table, with the other tables' progress "
          "(repetition, offset reached):")
    for event in live.events:
        others = ", ".join(
            f"{_label(names, q.cr)} {q.repetition + 1}:{q.offset}" for q in event.others
        ) or "-"
        verb = "starts" if event.kind == "first" else "ends  "
        banks = profiles[event.cr].ring_banks
        print(f"      cycle {event.cycle:>8}  {_label(names, event.cr):<20} {verb} "
              f"({banks} bank{'s' if banks != 1 else ''})   {others}")


def _check(run: TracedRun, bank_rows: int) -> list:
    return [] if run.declared is None else check_declared(run, bank_rows)


def _narrow(program: ProgramTrace) -> bool:
    """Whether the program ran with 128 B (narrow) XMEM rows."""
    from ipu_emu.ipu import xmem_row_size_bytes

    return program.state is not None and xmem_row_size_bytes(program.state) < 512


def report(run: TracedRun, bank_rows: int,
           restart_fraction: float = DEFAULT_RESTART_FRACTION) -> int:
    """Print the full report for one traced run. Returns 1 on declaration errors."""
    print(f"\n{run.kernel}  ({run.cycles} cycles, {bank_rows} rows/bank)")
    params = ", ".join(f"{k}={v}" for k, v in run.params.items() if _scalar(v))
    if params:
        print(f"  params: {params}")
    if any(_narrow(program) for program in run.session.programs):
        print("  NOTE: narrow mode. This tool is meant for wide mode: here a 512 B "
              "STR_POST_AAQ_REG also overwrites the 3 rows after the one it stores "
              "(interim POST_AAQ_REG width), which is not modeled.")
    for program in run.session.programs:
        _report_program(run, program, bank_rows, restart_fraction)

    if run.declared is None:
        print("\n  SPEC.tables: not declared (see the suggest: lines)")
        return 0
    findings = _check(run, bank_rows)
    if not findings:
        print("\n  SPEC.tables: matches this run")
        return 0
    print("\n  SPEC.tables:")
    for f in findings:
        print(f"    {f.severity.upper():<7} {f.table}: {f.message}")
    return int(any(f.severity == "error" for f in findings))


def _scalar(value) -> bool:
    return isinstance(value, (int, float, str, bool)) or (
        isinstance(value, tuple) and all(isinstance(v, (int, float)) for v in value))


def sweep(kernel: str, bank_rows: int, max_cycles: int | None,
          restart_fraction: float = DEFAULT_RESTART_FRACTION) -> int:
    """One row per case and table: parameters against the measured numbers."""
    rows = []
    status = 0
    for case_name, case in load_cases(kernel).items():
        try:
            run = run_traced(kernel, case, max_cycles=max_cycles)
        except Exception as exc:  # report the case, keep sweeping
            print(f"  {case_name}: FAILED -- {exc} (run --case {case_name} for its tables)",
                  file=sys.stderr)
            status = 1
            continue
        errors = {f.table for f in _check(run, bank_rows) if f.severity == "error"}
        status |= int(bool(errors))
        params = ", ".join(f"{k}={v}" for k, v in run.params.items() if _scalar(v))
        for program in run.session.programs:
            names = _names(run, program)
            declared = _declared(run, program)
            for cr, p in program.profiles(bank_rows, declared, restart_fraction).items():
                d = declared.get(cr)
                decl = (f"{d.jump_back}/{d.repetitions}/{d.rows}"
                        + (f"+{d.overhead_rows}" if d.overhead_rows else "")
                        if d is not None else "")
                mark = "!" if names.get(cr, f"cr{cr}") in errors else ""
                rows.append((case_name, program.index, _label(names, cr) + mark, _rows_text(p),
                             p.jump_back, p.repetitions, p.ring_banks, decl, params))
    print(f"\n{kernel}  (all cases, {bank_rows} rows/bank; ! = declaration error)")
    print(f"  {'case':<14} {'prog':>4} {'table':<21} {'rows':>6} {'jump_back':>9} "
          f"{'reps':>5} {'ring':>4} {'declared jb/reps/rows':>21}  params")
    for r in rows:
        print(f"  {r[0]:<14} {r[1]:>4} {r[2]:<21} {r[3]:>6} {r[4]:>9} {r[5]:>5} {r[6]:>4} "
              f"{r[7]:>21}  {r[8]}")
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="table_trace",
        description="Measure a kernel's XMEM tables (per base CR) for SPEC.tables.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--kernel", help="registered kernel name")
    target.add_argument("--all-family", metavar="OP",
                        help="report every kernel whose SPEC op is OP")
    parser.add_argument("--case", default="default", help="case name (default: default)")
    parser.add_argument("--all-cases", action="store_true",
                        help="tabulate every case of --kernel against its parameters")
    parser.add_argument("--bank-rows", type=int, default=BANK_ROWS,
                        help=f"rows per bank for bank-level numbers (default {BANK_ROWS})")
    parser.add_argument("--restart-fraction", type=float, default=DEFAULT_RESTART_FRACTION,
                        help="for undeclared tables, a backward jump across this fraction "
                             f"of the rows starts a repetition (default {DEFAULT_RESTART_FRACTION})")
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args(argv)
    if args.bank_rows <= 0:
        parser.error("--bank-rows must be positive")
    if not 0 < args.restart_fraction <= 1:
        parser.error("--restart-fraction must be in (0, 1]")

    if args.all_cases:
        if not args.kernel:
            parser.error("--all-cases needs --kernel")
        return sweep(args.kernel, args.bank_rows, args.max_cycles, args.restart_fraction)

    if args.all_family:
        names = sorted(s.name for s in kernels(args.all_family))
        if not names:
            print(f"no kernels with op {args.all_family!r}", file=sys.stderr)
            return 1
    else:
        names = [args.kernel]

    status = 0
    for name in names:
        try:
            cases = load_cases(name)
            if args.case not in cases:
                raise ValueError(f"no case {args.case!r}; have {sorted(cases)}")
            run = run_traced(name, cases[args.case], max_cycles=args.max_cycles)
        except TracedRunFailed as failed:
            # The kernel failed, but what it accessed until then is still shown.
            if failed.run is not None:
                report(failed.run, args.bank_rows, args.restart_fraction)
            print(f"\n{name}: FAILED -- {failed}", file=sys.stderr)
            status = 1
            continue
        except Exception as exc:  # one kernel failing must not hide the others
            print(f"\n{name}: FAILED -- {exc}", file=sys.stderr)
            status = 1
            continue
        status |= report(run, args.bank_rows, args.restart_fraction)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
