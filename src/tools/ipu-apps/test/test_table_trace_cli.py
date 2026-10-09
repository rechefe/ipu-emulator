"""The table_trace CLI and the runner's --tables: report, declaration check, exit status."""

import re

import pytest

from ipu_emu.table_trace import Table

from ipu_apps import table_trace_cli
from ipu_apps.kernel_registry import runner
from ipu_apps.kernel_registry import tables as registry_tables


def _row(out, table):
    """The summary-table row for ``table`` as {column: value}."""
    line = next(line for line in out.splitlines() if line.strip().startswith(table))
    fields = re.split(r"\s{2,}", line.strip())
    return dict(zip(["table", "type", "rows", "jump_back", "reps"], fields))


def test_declared_kernel_matches_at_the_real_bank_size(capsys):
    assert table_trace_cli.main(["--kernel", "layernorm_64x192"]) == 0
    out = capsys.readouterr().out
    # What the table shows is what the check checked.
    assert _row(out, "centered (CR7)")["jump_back"] == "1"
    assert _row(out, "centered (CR7)")["reps"] == "4"
    assert _row(out, "data (CR13)")["reps"] == "2"
    assert _row(out, "data (CR13)")["rows"] == "192+1"      # declared overhead row
    assert "SPEC.tables: matches this run" in out
    assert "last access (release point)" in out
    assert "LDR_CYCLIC_MULT_REG LR2, CR7, LR0" in out    # as the debugger shows it


def test_mismatch_is_reported_and_fails(capsys, monkeypatch):
    monkeypatch.setattr(registry_tables, "declared_tables", lambda kernel, params: {
        "input": Table(cr=12, rows=128),
        "weights": Table(cr=11, repetitions=1, rows=128),
        "output": Table(cr=2, rows=128),
    })
    assert table_trace_cli.main(["--kernel", "matmul_128x128"]) == 1
    out = capsys.readouterr().out
    assert "ERROR   weights: 128 repetitions, declared 1" in out


def test_undeclared_overhead_is_highlighted_and_fails(capsys, monkeypatch):
    real = registry_tables.declared_tables

    def without_overhead(kernel, params):
        return {name: Table(t.cr, t.jump_back, t.repetitions, t.rows, 0, t.program)
                for name, t in real(kernel, params).items()}

    monkeypatch.setattr(registry_tables, "declared_tables", without_overhead)
    assert table_trace_cli.main(["--kernel", "layernorm_64x192"]) == 1
    out = capsys.readouterr().out
    assert _row(out, "centered (CR7)")["rows"] == "192+0!"   # as declared, with a gap
    assert "!! GAP: offsets 192..192 accessed but not declared" in out
    assert "offset 192: 3x, first offset 192 @ cycle 1560, pc 38" in out
    assert "ERROR   centered: 3 accesses outside its declared rows=192" in out


def test_overlapping_tables_fail(capsys, monkeypatch):
    # The harness leaves one row after the weights for their last prefetch;
    # declaring two makes the weights' allocation run onto the output's first
    # row: one table, one ID.
    monkeypatch.setattr(registry_tables, "declared_tables", lambda kernel, params: {
        "input": Table(cr=12, rows=128),
        "weights": Table(cr=11, repetitions=128, rows=128, overhead_rows=2),
        "output": Table(cr=2, rows=128),
    })
    assert table_trace_cli.main(["--kernel", "matmul_128x128"]) == 1
    out = capsys.readouterr().out
    assert "!! its allocated rows overlap output (CR2) at XMEM rows 257..257" in out
    assert "ERROR   weights: program 0: allocated rows 128..257" in out


def test_runner_reports_fail_when_tables_disagree(capsys, monkeypatch):
    monkeypatch.setattr(registry_tables, "declared_tables", lambda kernel, params: {
        "input": Table(cr=12, rows=128),
        "weights": Table(cr=11, repetitions=1, rows=128, overhead_rows=1),
        "output": Table(cr=2, rows=128),
    })
    assert runner.main(["--kernel", "matmul_128x128", "--tables"]) == 1
    out = capsys.readouterr().out
    assert "matmul_128x128/default: FAIL" in out
    assert "PASS" not in out


def test_failing_kernel_still_reports_its_tables(capsys):
    # Too few cycles: the kernel cannot finish, but what it touched is shown.
    assert table_trace_cli.main(["--kernel", "matmul_128x128", "--max-cycles", "500"]) == 1
    captured = capsys.readouterr()
    assert "weights (CR11)" in captured.out
    assert "FAILED" in captured.err


def test_narrow_mode_is_noted(capsys):
    assert table_trace_cli.main(["--kernel", "fully_connected", "--case", "int8"]) == 0
    assert "NOTE: narrow mode" in capsys.readouterr().out


def test_wide_mode_has_no_narrow_note(capsys):
    assert table_trace_cli.main(["--kernel", "matmul_128x128"]) == 0
    assert "narrow mode" not in capsys.readouterr().out


def test_undeclared_kernel_gets_suggestions(capsys, monkeypatch):
    # Every kernel declares its tables now; drop one declaration.
    monkeypatch.setattr(registry_tables, "declared_tables", lambda kernel, params: None)
    assert table_trace_cli.main(["--kernel", "softmax_rows_partial"]) == 0
    out = capsys.readouterr().out
    assert "suggest:  Table(cr=" in out
    assert "declare them as overhead_rows instead" in out
    assert "SPEC.tables: not declared" in out


def test_each_program_of_a_multi_stage_kernel_is_reported(capsys):
    assert table_trace_cli.main(["--kernel", "depthwise_conv_stride2_128"]) == 0
    out = capsys.readouterr().out
    assert "program 0" in out and "program 1" in out


def test_all_cases_checks_declarations(capsys, monkeypatch):
    assert table_trace_cli.main(["--kernel", "matmul_128x128", "--all-cases"]) == 0
    monkeypatch.setattr(registry_tables, "declared_tables", lambda kernel, params: {
        "input": Table(cr=12), "weights": Table(cr=11), "output": Table(cr=2)})
    assert table_trace_cli.main(["--kernel", "matmul_128x128", "--all-cases"]) == 1
    assert "weights (CR11)!" in capsys.readouterr().out


def test_runner_tables_flag(capsys):
    assert runner.main(["--kernel", "matmul_128x128", "--tables"]) == 0
    out = capsys.readouterr().out
    assert "weights (CR11)" in out
    assert "SPEC.tables: matches this run" in out
    assert "matmul_128x128/default: PASS" in out


def test_unknown_case_fails(capsys):
    assert table_trace_cli.main(["--kernel", "matmul_128x128", "--case", "nope"]) == 1
    assert "no case 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--all-family", "matmul", "--all-cases"],
    ["--kernel", "matmul_128x128", "--restart-fraction", "0"],
    ["--kernel", "matmul_128x128", "--bank-rows", "0"],
])
def test_bad_arguments(argv):
    with pytest.raises(SystemExit):
        table_trace_cli.main(argv)
