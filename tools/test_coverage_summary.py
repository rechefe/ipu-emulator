"""Tests for tools/coverage_summary.py (the nightly job's coverage table)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import coverage_summary as cs  # noqa: E402

_REPORT = """\
SF:src/tools/ipu-emu/src/ipu_emu/ipu.py
DA:1,1
DA:2,0
DA:3,4
end_of_record
SF:src/tools/ipu-emu/src/ipu_emu/ipu.py
DA:2,1
DA:5,0
end_of_record
SF:src/tools/ipu-as/src/ipu_as/lark_tree.py
DA:10,0
LH:0
LF:1
end_of_record
SF:docs/update_release_log.py
LH:3
LF:4
end_of_record
"""


def test_package_of():
    assert cs.package_of("src/tools/ipu-emu/src/ipu_emu/ipu.py") == "ipu_emu"
    assert cs.package_of("/x/execroot/_main/src/tools/ipu-isa/src/ipu_isa/a/b.py") == "ipu_isa"
    assert cs.package_of("src/tools/ipu-harness/test/conftest.py") == "src/tools/ipu-harness"
    assert cs.package_of("docs/update_release_log.py") == "docs"
    assert cs.package_of("setup.py") == "(root)"


def test_records_for_one_file_are_merged_by_line():
    per_file = cs.parse_lcov(_REPORT)
    ipu = per_file["src/tools/ipu-emu/src/ipu_emu/ipu.py"]
    # Lines 1, 2, 3, 5; line 2 is hit by the second record only.
    assert (ipu.hit, ipu.found) == (3, 4)


def test_lh_lf_used_without_da():
    per_file = cs.parse_lcov(_REPORT)
    log = per_file["docs/update_release_log.py"]
    assert (log.hit, log.found) == (3, 4)
    # DA records take precedence over LH/LF in the same record.
    lark = per_file["src/tools/ipu-as/src/ipu_as/lark_tree.py"]
    assert (lark.hit, lark.found) == (0, 1)


def test_markdown_table(tmp_path, capsys):
    report = tmp_path / "r.dat"
    report.write_text(_REPORT)
    assert cs.main([str(report), "--require", "ipu_emu"]) == 0
    out = capsys.readouterr().out
    assert "| `ipu_emu` | 3 | 4 | 75.0 % |" in out
    assert "| **Total** | 6 | 9 | 66.7 % |" in out


def test_required_package_without_coverage_fails(tmp_path, capsys):
    report = tmp_path / "r.dat"
    report.write_text(_REPORT)
    assert cs.main([str(report), "--require", "ipu_as"]) == 1
    assert "package ipu_as has no covered line" in capsys.readouterr().err
    assert cs.main([str(report), "--require", "ipu_isa"]) == 1


def test_missing_or_empty_report_fails(tmp_path):
    assert cs.main([str(tmp_path / "absent.dat")]) == 1
    empty = tmp_path / "empty.dat"
    empty.write_text("")
    assert cs.main([str(empty)]) == 1
    no_lines = tmp_path / "nolines.dat"
    no_lines.write_text("SF:a/b.py\nend_of_record\n")
    assert cs.main([str(no_lines)]) == 1
