"""The entry of every ipu_kernel() target (E06-13), run as Bazel runs it.

``main(toml, shim, tier, *args)`` with ``BAZEL_TEST`` set runs the generated
tests through the real rules_python_pytest shim; with
``BUILD_WORKING_DIRECTORY`` set it runs the CLI. Each test works on a copy of
//kernels/elementwise/scale, so it can plant files.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ipu_harness import kernel_test
from ipu_harness.kernel_bin import default_bin

#: Bazel runs the test from the runfiles root of the main repository.
SCALE = Path.cwd() / "kernels" / "elementwise" / "scale"
SHIM = os.path.abspath(os.environ["PYTEST_SHIM"])

#: Bazel's variables of a test run, cleared so each test sets its own.
_BAZEL = ("TESTBRIDGE_TEST_ONLY", "TEST_SHARD_INDEX", "TEST_TOTAL_SHARDS", "XML_OUTPUT_FILE")


@pytest.fixture
def scale(tmp_path, monkeypatch):
    directory = tmp_path / "elementwise" / "scale"
    shutil.copytree(SCALE, directory)
    for name in _BAZEL:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BAZEL_TEST", "1")
    monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)
    monkeypatch.delenv("TEST_SHARD_STATUS_FILE", raising=False)
    return directory


def _test(directory: Path, tier: str = "default", *args: str) -> int:
    return kernel_test.main([str(directory / "kernel.toml"), SHIM, tier, "-v", *args])


def test_every_case_in_every_format(scale, capsys):
    assert _test(scale) == 0
    out = capsys.readouterr().out
    for case in ("default", "one_row", "mine"):
        for fmt in ("fp32", "int8", "e4"):
            assert f"test_case[{case}-{fmt}] PASSED" in out
    for rows in (1, 7, 33, 128):
        for fmt in ("fp32", "e4"):
            assert f"test_case[sweep[rows={rows}]-{fmt}] PASSED" in out
    assert out.count(" PASSED") == 17


def test_a_wrong_program_names_the_case_format_index_and_tier(scale, capsys):
    asm = scale / "scale.asm"
    asm.write_text(asm.read_text().replace("MULT.VE LR0 {{ cr.s }}", "MULT.VE LR0 CR1"))  # y = x
    assert _test(scale) != 0
    out = capsys.readouterr().out
    assert "test_case[default-int8] FAILED" in out
    assert "scale/default/int8: FAIL" in out and "y: FAIL at EXACT" in out
    assert "worst at [" in out


def test_the_folders_own_tests_run_in_the_default_target(scale, capsys):
    (scale / "helpers.py").write_text("WANT = ['default', 'one_row']\n")
    (scale / "test_extra.py").write_text(
        "from helpers import WANT\n\n\n"
        "def test_extra(kernel, cases, run_case):\n"
        "    assert kernel.name == 'scale'\n"
        "    assert [c.name for c in cases][:2] == WANT\n"
        "    assert run_case(cases[1], cases[1].formats[0]).passed\n"
    )
    assert _test(scale) == 0
    assert "test_extra.py::test_extra PASSED" in capsys.readouterr().out
    assert _test(scale, "nightly") == 0  # not in the nightly target
    assert "test_extra" not in capsys.readouterr().out


def test_nightly_cases_run_only_in_the_nightly_target(scale, capsys):
    toml = scale / "kernel.toml"
    toml.write_text(toml.read_text() + '\n[cases.big]\nrows = 64\ns = 2.0\ntier = "nightly"\n')
    assert _test(scale) == 0
    assert "test_case[big-" not in capsys.readouterr().out
    assert _test(scale, "nightly") == 0
    out = capsys.readouterr().out
    assert "test_case[big-fp32] PASSED" in out and "test_case[default-" not in out


def test_shards_split_the_runs(scale, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_TOTAL_SHARDS", "3")
    monkeypatch.setenv("TEST_SHARD_STATUS_FILE", str(tmp_path / "shard"))
    passed = 0
    for shard in range(3):
        monkeypatch.setenv("TEST_SHARD_INDEX", str(shard))
        assert _test(scale) == 0
        passed += capsys.readouterr().out.count(" PASSED")
    assert passed == 17


def test_a_test_filter_selects_runs(scale, capsys, monkeypatch):
    monkeypatch.setenv("TESTBRIDGE_TEST_ONLY", "one_row")
    assert _test(scale) == 0
    out = capsys.readouterr().out
    assert out.count(" PASSED") == 3 and "test_case[default-" not in out


def test_bazel_run_runs_the_cli(scale, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))
    toml = str(scale / "kernel.toml")
    assert kernel_test.main([toml, SHIM, "default", "--case", "one_row", "--format", "e4"]) == 0
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("scale/one_row/e4: PASS  cycles=") and "MAC util=" in line


# --- a registry.toml entry (E06-17), on the fixture family test/scales ---------------

SCALES = Path.cwd() / "src" / "tools" / "ipu-harness" / "test" / "scales" / "registry.toml"


def test_an_entry_runs_the_template_cases_with_its_values(scale, capsys):
    # scale_r8_fp32: rows = 8, formats ["fp32"], cases ["default"].
    assert kernel_test.main([f"{SCALES}::scale_r8_fp32", SHIM, "default", "-v"]) == 0
    out = capsys.readouterr().out
    assert "test_case[default-fp32] PASSED" in out and out.count(" PASSED") == 1


def test_bazel_run_of_an_entry_runs_the_cli(scale, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))
    assert kernel_test.main([f"{SCALES}::scale_r2", SHIM, "default", "--case", "negative"]) == 0
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("scale_r2/negative/fp32: PASS")


@pytest.mark.parametrize(
    ("target", "kernel"),
    [
        (SCALE / "scale.bin", str(SCALE)),
        (SCALES.parent / "scale_r2.bin", f"{SCALES}::scale_r2"),
    ],
)
def test_the_assemble_targets_write_the_default_case_program(target, kernel):
    # The genrules' outputs sit beside the sources in the runfiles tree.
    assert target.read_bytes() == default_bin(kernel)
    assert target.read_bytes()  # a program, not an empty file
