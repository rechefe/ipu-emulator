"""Tests for the per-table XMEM access tracer (ipu_emu.table_trace)."""

import pytest

from ipu_as.lark_tree import assemble

from ipu_emu.emulator import load_program, run_until_complete
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState
from ipu_emu.prototyping import build_enabled
from ipu_emu.table_trace import (
    ProgramTrace, Table, TraceSession, check_tables, liveness, trace_tables,
)


class _Engine:
    """Stands in for an ``Ipu`` when feeding synthetic accesses."""

    def __init__(self, cycles):
        self.state = None
        self.cycles_issued = cycles


def _program(*streams, cycles=None):
    """A ProgramTrace from ``(cr, kind, offsets)`` streams, interleaved in
    order: one access per cycle, streams taking turns."""
    iters = [(cr, kind, iter(offsets)) for cr, kind, offsets in streams]
    accesses = []
    while iters:
        for item in list(iters):
            cr, kind, it = item
            offset = next(it, None)
            if offset is None:
                iters.remove(item)
            else:
                accesses.append((cr, kind, offset))
    program = ProgramTrace(0, _Engine(cycles if cycles is not None else len(accesses)))
    for cycle, (cr, kind, offset) in enumerate(accesses):
        program.record(cycle, cycle, "LDR" if kind == "read" else "STR",
                       kind, cr, 100 * cr, 0, offset)
    program.finish()
    return program


def _one(offsets, *, declared=None, bank_rows=1024, kind="read"):
    """Profile of a single table (CR2) read at ``offsets``."""
    program = _program((2, kind, offsets))
    tables = {2: declared} if declared is not None else None
    return program.profiles(bank_rows, tables)[2]


def _check(offsets, table, kind="read"):
    return check_tables(TraceSession(programs=[_program((table.cr, kind, offsets))]),
                        {"t": table})


def _stencil(end):
    """A 3-row window walking forward, newest row first: each step reaches
    two rows below the high-water mark it just set."""
    return [o for i in range(2, end) for o in (i, i - 2, i - 1)]


# -- repetitions and jump_back, undeclared (suggestion rule) -------------------


def test_forward_stream():
    p = _one(range(40))
    assert (p.jump_back, p.repetitions, p.rows) == (0, 1, 40)
    assert p.suggested_type == "read_only"


def test_sliding_window_reach_and_where_it_happens():
    p = _one(_stencil(40))
    assert (p.jump_back, p.repetitions) == (2, 1)
    assert p.worst.reach_from.offset - p.worst.reach_to.offset == 2


def test_full_rereads_are_repetitions():
    p = _one(list(range(32)) * 3)
    assert (p.repetitions, p.jump_back) == (3, 0)


def test_restart_need_not_return_to_offset_zero():
    p = _one(list(range(32)) + list(range(1, 32)))
    assert (p.repetitions, p.jump_back) == (2, 0)


def test_decimated_phases_are_repetitions():
    p = _one(list(range(0, 64, 2)) + list(range(1, 64, 2)))
    assert (p.repetitions, p.jump_back) == (2, 0)


def test_reverse_walk_is_reach_not_restarts():
    # Small steps back add up to a full-table reach; no single jump restarts.
    p = _one(range(31, -1, -1))
    assert (p.repetitions, p.jump_back) == (1, 31)


def test_restart_fraction_is_configurable():
    offsets = list(range(32)) + list(range(16, 32))      # jump back of 31 -> 16 = 15
    program = _program((2, "read", offsets))
    assert program.profiles(restart_fraction=0.9)[2].repetitions == 1
    assert program.profiles(restart_fraction=0.4)[2].repetitions == 2


# -- the same, under a declaration ------------------------------------------------


def test_declared_jump_back_splits_repetitions():
    offsets = [0, 1, 2, 3, 0, 1, 2, 3]
    assert _one(offsets, declared=Table(cr=2, jump_back=0)).repetitions == 2
    assert _one(offsets, declared=Table(cr=2, jump_back=3)).repetitions == 1


def test_header_row_revisited_mid_stream_needs_a_large_enough_jump_back():
    offsets = [o for i in range(1, 40) for o in (0, i)]       # header, then data row i
    assert _check(offsets, Table(cr=2, jump_back=39, rows=40)) == []
    (finding,) = _check(offsets, Table(cr=2, jump_back=0, rows=40))
    assert finding.severity == "error"
    assert "repetitions, declared 1" in finding.message
    assert "jump back of" in finding.message


def test_reach_beyond_jump_back_without_any_restart_is_an_error():
    # The reverse walk never jumps more than 1, so it stays one repetition,
    # but its reach is the whole table.
    (finding,) = _check(range(31, -1, -1), Table(cr=2, jump_back=1, rows=32))
    assert finding.severity == "error"
    assert "reaches back 31 rows, declared jump_back=1" in finding.message
    assert "pc" in finding.message


def test_table_away_from_offset_zero_has_no_false_reach():
    # Rows 16..31 read three times: plain repetitions.
    assert _check(list(range(16, 32)) * 3, Table(cr=2, repetitions=3, rows=32)) == []


def test_matching_declaration_is_clean():
    assert _check(_stencil(40), Table(cr=2, jump_back=2, rows=40)) == []


def test_repetitions_mismatch():
    (finding,) = _check(list(range(32)) * 3, Table(cr=2, repetitions=2, rows=32))
    assert finding.severity == "error"
    assert finding.message.startswith("3 repetitions, declared 2")
    (finding,) = _check(list(range(32)) * 3, Table(cr=2, repetitions=4, rows=32))
    assert finding.message.startswith("3 repetitions, declared 4")


def test_one_row_table_repetitions_are_not_checked():
    # Every access is row 0: a second traversal cannot be seen, so a formula
    # that says 2 (true for larger sizes) is not contradicted.
    assert _check([0, 0, 0], Table(cr=2, repetitions=2, rows=1)) == []
    (finding,) = _check([0, 1, 0, 1], Table(cr=2, repetitions=3, rows=2))
    assert finding.message.startswith("2 repetitions, declared 3")


# -- overreads ------------------------------------------------------------------


def test_undeclared_overhead_fails_with_the_gap_and_every_access():
    # A 16-row table re-read twice, each pass prefetching rows 16 and 17.
    offsets = list(range(18)) * 2
    (finding,) = _check(offsets, Table(cr=2, repetitions=2, rows=16))
    assert finding.severity == "error"
    assert "4 accesses outside its declared rows=16 + overhead_rows=0" in finding.message
    assert "GAP: offsets 16..17" in finding.message
    assert "overhead_rows=2" in finding.message
    assert "offset 16: 2x" in finding.message and "offset 17: 2x" in finding.message


def test_partly_declared_overhead_still_fails_on_the_rest():
    (finding,) = _check(list(range(18)) * 2, Table(cr=2, repetitions=2, rows=16,
                                                   overhead_rows=1))
    assert finding.severity == "error"
    assert "GAP: offsets 17..17" in finding.message


def test_declared_overhead_is_allocated():
    offsets = list(range(18)) * 2
    table = Table(cr=2, repetitions=2, rows=16, overhead_rows=2)
    assert _check(offsets, table) == []
    p = _one(offsets, declared=table, bank_rows=16)
    assert p.allocated_rows == 18
    assert p.banks == 2          # the overhead rows spill into a second bank
    assert p.ring_banks == 2


def test_unused_overhead_is_a_warning():
    (finding,) = _check(range(16), Table(cr=2, rows=16, overhead_rows=1))
    assert finding.severity == "warning"
    assert "overhead_rows=1" in finding.message


def test_access_below_the_base_fails():
    (finding,) = _check([-1, 0, 1], Table(cr=2, jump_back=1, rows=2))
    assert finding.severity == "error"
    assert "below the table's base" in finding.message


def test_nothing_widens_a_declaration_by_itself():
    table = Table(cr=2, rows=16)
    _check(list(range(18)), table)
    assert (table.rows, table.overhead_rows) == (16, 0)


# -- banks ------------------------------------------------------------------------


def test_streaming_table_needs_two_banks():
    p = _one(_stencil(64), bank_rows=16)
    assert (p.banks, p.ring_banks, p.slots) == (4, 2, 2)
    assert not p.resident


def test_repeated_table_stays_resident_until_its_last_repetition():
    p = _one(list(range(64)) * 2, bank_rows=16)
    assert (p.banks, p.ring_banks) == (4, 4)
    assert p.resident


def test_slots_avoid_bank_collisions():
    # Rows 0..15 then 32..63 at 16 rows/bank: banks {0, 2, 3} live at once
    # with nothing released (bank 1 is never visited), and 0 and 3 collide
    # in a ring of 3.
    p = _one(list(range(16)) + list(range(32, 64)), bank_rows=16)
    assert p.ring_banks == 3
    assert p.slots == 4


def test_liveness_and_idle_tail():
    p = _program((2, "read", range(10)), cycles=100).profiles()[2]
    assert (p.first.cycle, p.last.cycle, p.idle_tail) == (0, 9, 90)


# -- types and writes ----------------------------------------------------------------


def test_type_and_write_order():
    out_of_order = _program((4, "write", [0, 2, 1, 3, 3])).profiles()[4]
    assert out_of_order.suggested_type == "write_only"
    assert (out_of_order.out_of_order_writes, out_of_order.rows_rewritten) == (1, 1)

    raw = _program((5, "write", [0, 1]), (5, "read", [0, 1])).profiles()[5]
    assert raw.suggested_type == "read_after_write"

    both = _program((6, "read", [0]), (6, "write", [0])).profiles()[6]
    assert both.suggested_type == "read+write"


def test_type_counts_overhead_reads():
    # Written in its data rows, read only in its overhead row: still read.
    program = _program((5, "write", [0, 1]), (5, "read", [2]))
    p = program.profiles(declared={5: Table(cr=5, rows=2, overhead_rows=1)})[5]
    assert p.suggested_type != "write_only"


def test_type_ignores_overread_rows():
    # Written rows 0..1 read back, plus an overread of row 2 never written.
    program = _program((5, "write", [0, 1]), (5, "read", [0, 1, 2]))
    p = program.profiles(declared={5: Table(cr=5, rows=2, overhead_rows=1)})[5]
    assert p.suggested_type == "read_after_write"


def test_changed_base_is_reported():
    program = ProgramTrace(0, _Engine(2))
    program.record(0, 0, "LDR", "read", 2, 10, 0, 0)
    program.record(1, 1, "LDR", "read", 2, 20, 0, 0)
    assert program.profiles()[2].bases == (10, 20)
    (finding,) = check_tables(TraceSession(programs=[program]), {"t": Table(cr=2, rows=1)})
    assert "changed while in use" in finding.message


def test_negative_offset_is_signed():
    program = ProgramTrace(0, _Engine(1))
    program.record(0, 0, "LDR", "read", 2, 10, 0, 0xFFFFFFFF)
    assert program.profiles()[2].min_offset == -1


def test_rows_are_required_in_a_declaration():
    (finding,) = _check(list(range(10)) + [500], Table(cr=2))
    assert finding.severity == "error"
    assert "declares no rows" in finding.message and "offset 500" in finding.message


def test_whole_run_table_is_never_released_and_skips_the_jump_rules():
    offsets = list(range(64)) * 3 + list(range(63, -1, -1))
    table = Table(cr=2, rows=64, whole_run=True)
    assert _check(offsets, table) == []
    p = _one(offsets, declared=table, bank_rows=16)
    assert (p.repetitions, p.ring_banks, p.banks) == (1, 4, 4)
    # Its rows are still checked.
    (finding,) = _check(offsets + [64], table)
    assert "GAP: offsets 64..64" in finding.message


def test_small_undeclared_table_reread_is_a_repetition():
    # 8 rows read twice: the jump back is 7, the largest possible.
    p = _one(list(range(8)) * 2)
    assert (p.repetitions, p.jump_back) == (2, 0)


def test_out_of_bounds_access_is_reported_once():
    # Below the base: one error, not also an extra repetition.
    (finding,) = _check([0, 1, -1, 2], Table(cr=2, jump_back=0, rows=3))
    assert "below the table's base" in finding.message


def test_more_banks_than_xmem_has_is_a_warning():
    # Two whole-run tables of 9 banks each at 16 rows/bank: 18 banks at once
    # (CR2 at row 200 and CR5 at row 500, so they do not overlap).
    program = _program((2, "read", range(144)), (5, "read", range(144)))
    tables = {"a": Table(cr=2, rows=144, whole_run=True),
              "b": Table(cr=5, rows=144, whole_run=True)}
    findings = check_tables(TraceSession(programs=[program]), tables, bank_rows=16)
    assert [(f.severity, f.table) for f in findings] == [("warning", "*")]
    assert "needs 18 banks at once; XMEM has 16" in findings[0].message


def test_ring_beyond_xmem_num_of_banks_is_a_warning():
    findings = _check_banks_only(range(16 * 16), Table(cr=2, rows=256, whole_run=True))
    assert ("warning", "t") in {(f.severity, f.table) for f in findings}


def _check_banks_only(offsets, table):
    program = _program((table.cr, "read", offsets))
    return check_tables(TraceSession(programs=[program]), {"t": table}, bank_rows=16)


# -- across tables -------------------------------------------------------------


def test_liveness_peak_and_order():
    # cr2 streams in first, then cr3 is written after cr2 is finished.
    program = _program((2, "read", range(32)))
    for i in range(32):
        program.record(100 + i, 50, "STR", "write", 3, 300, 0, i)
    program.cycles = 200
    live = liveness(program.profiles(bank_rows=16))
    assert live.peak_banks == 2
    assert [(e.cr, e.kind) for e in live.events] == [
        (2, "first"), (2, "last"), (3, "first"), (3, "last")]


def test_single_access_table_starts_before_it_ends():
    live = liveness(_program((2, "read", [0])).profiles())
    assert [e.kind for e in live.events] == ["first", "last"]


def test_end_is_expressed_in_other_tables_progress():
    # Weights (cr2) read alongside the output (cr3); when the weights end the
    # output's high-water is offset 4, in its first repetition.
    program = _program((2, "read", range(6)), (3, "write", range(12)))
    live = liveness(program.profiles())
    (end,) = [e for e in live.events if e.cr == 2 and e.kind == "last"]
    (progress,) = end.others
    assert (progress.cr, progress.repetition, progress.offset) == (3, 0, 4)


# -- declarations, other cases ----------------------------------------------------


def test_undeclared_cr_and_flat_base():
    session = TraceSession(programs=[_program((0, "read", [5]), (2, "read", [0]))])
    findings = check_tables(session, {"data": Table(cr=2, rows=1)})
    assert [(f.severity, f.table) for f in findings] == [("error", "cr0")]
    assert "flat base" in findings[0].message


def test_table_declared_on_cr0_is_an_error():
    session = TraceSession(programs=[_program((0, "read", [5]))])
    findings = check_tables(session, {"data": Table(cr=0, rows=6)})
    assert ("error", "data") in {(f.severity, f.table) for f in findings}
    assert any("declared on CR0" in f.message for f in findings)


def test_unused_and_shared_cr():
    session = TraceSession(programs=[_program((2, "read", range(10)))])
    findings = check_tables(session, {
        "data": Table(cr=2, rows=10),
        "bias": Table(cr=9),
        "alias": Table(cr=2),
    })
    messages = {(f.severity, f.table): f.message for f in findings}
    assert "shares CR2" in messages[("error", "alias")]
    assert ("warning", "bias") in messages
    assert len(findings) == 2


def test_overlapping_tables_are_an_error():
    # CR2's base is row 200, CR3's row 300: 100 rows of data plus one
    # overhead row run onto CR3's first row.
    session = TraceSession(programs=[_program((2, "read", range(101)), (3, "read", range(4)))])
    findings = check_tables(session, {
        "a": Table(cr=2, rows=100, overhead_rows=1),
        "b": Table(cr=3, rows=4),
    })
    assert [(f.severity, f.table) for f in findings] == [("error", "a")]
    assert "overlap 'b' (CR3, rows 300..303) at XMEM rows 300..300" in findings[0].message


def test_adjacent_tables_do_not_overlap():
    session = TraceSession(programs=[_program((2, "read", range(100)), (3, "read", range(4)))])
    assert check_tables(session, {"a": Table(cr=2, rows=100), "b": Table(cr=3, rows=4)}) == []


def test_tables_for_a_program_that_never_ran():
    session = TraceSession(programs=[_program((2, "read", [0]))])
    findings = check_tables(session, {"t": Table(cr=2), "s": Table(cr=3, program=1)})
    assert ("error", "*") in {(f.severity, f.table) for f in findings}


def test_empty_session_has_nothing_to_check():
    assert check_tables(TraceSession(), {}) == []


# -- end to end through the engine ---------------------------------------------


_PROGRAM = """\
SET lr0 cr0;;
LDR_MULT_REG r0 lr0 cr2;;
INC lr0 1;;
LDR_MULT_REG r0 lr0 cr2;;
LDR_MULT_REG r1 lr0 cr3;;
STR_POST_AAQ_REG lr0 cr4;;
BKPT;;
"""


def _state(prototyping=None):
    state = IpuState(prototyping=prototyping)
    # cr2 and cr3 hold the same base: only the CR index tells them apart.
    for cr, value in {2: 10, 3: 10, 4: 50}.items():
        state.regfile.set_cr(cr, value)
    load_program(state, [decode_instruction_word(w) for w in assemble(_PROGRAM)])
    return state


def test_engine_attributes_accesses_by_cr_not_by_value():
    state = _state()
    state.xmem.write_address(0, b"\x00" * 128)     # host write: not the kernel's
    with trace_tables() as session:
        run_until_complete(state)
    (program,) = session.programs
    reads = [(a.cr, a.base, a.lr, a.offset) for a in program.accesses if a.kind == "read"]
    assert reads == [(2, 10, 0, 0), (2, 10, 0, 1), (3, 10, 0, 1)]
    assert [a.cycle for a in program.accesses if a.kind == "read"] == [1, 3, 4]
    assert program.cycles == state.stats.total_cycles


def test_engine_store_is_one_row_whatever_the_width():
    state = _state()
    with trace_tables() as session:
        run_until_complete(state)
    writes = [a.offset for a in session.programs[0].accesses if a.kind == "write"]
    # Tables are addressed in rows; the 512 B store is one row access.
    assert writes == [1]


@pytest.mark.skipif(not build_enabled(), reason="needs --define=ipu_proto=1")
def test_engine_tracing_bypasses_prototyping_fast_path():
    state = _state(prototyping=True)
    with trace_tables() as session:
        run_until_complete(state)
    assert len(session.programs[0].accesses) == 4


def test_each_program_recorded_separately_and_only_in_scope():
    first, second, outside = _state(), _state(), _state()
    with trace_tables() as session:
        run_until_complete(first)
        run_until_complete(second)
    run_until_complete(outside)
    assert [p.index for p in session.programs] == [0, 1]
    assert all(len(p.accesses) == 4 for p in session.programs)
