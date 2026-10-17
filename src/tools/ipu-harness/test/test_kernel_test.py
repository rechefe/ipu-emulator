"""The entry of every ipu_kernel() target (E06-13), run as Bazel runs it.

``main(toml, shim, tier, *args)`` with ``BAZEL_TEST`` set runs the generated
tests through the shim; with ``BUILD_WORKING_DIRECTORY`` set it runs the CLI.
The shim here is a stand-in for rules_python_pytest's: it runs pytest on its
arguments.
"""

from __future__ import annotations

import pytest
from scale_kernel import write_scale

from ipu_harness import kernel_test

SHIM = "import sys\n\nimport pytest\n\nsys.exit(pytest.main(sys.argv[1:]))\n"


@pytest.fixture
def scale(tmp_path):
    return write_scale(tmp_path, with_reference=False)


@pytest.fixture
def shim(tmp_path):
    path = tmp_path / "shim.py"
    path.write_text(SHIM)
    return str(path)


def _test(monkeypatch, manifest, shim, tier="default", *args):
    monkeypatch.setenv("BAZEL_TEST", "1")
    monkeypatch.delenv("BUILD_WORKING_DIRECTORY", raising=False)
    return kernel_test.main([str(manifest.directory / "kernel.toml"), shim, tier, "-v", *args])


def test_every_case_in_every_format(scale, shim, monkeypatch, capsys):
    assert _test(monkeypatch, scale, shim) == 0
    out = capsys.readouterr().out
    for case in ("default", "one_row", "mine"):
        for fmt in ("fp32", "int8", "e4"):
            assert f"test_case[{case}-{fmt}] PASSED" in out


def test_a_wrong_program_names_the_case_format_index_and_tier(scale, shim, monkeypatch, capsys):
    asm = scale.directory / "scale.asm"
    asm.write_text(asm.read_text().replace("MULT.VE LR0 {{ cr.s }}", "MULT.VE LR0 CR1"))  # y = x
    assert _test(monkeypatch, scale, shim) != 0
    out = capsys.readouterr().out
    assert "test_case[default-int8] FAILED" in out
    assert "scale/default/int8: FAIL" in out and "y: FAIL at EXACT" in out
    assert "worst at [" in out


def test_the_folders_own_tests_run_in_the_target(scale, shim, monkeypatch, capsys):
    (scale.directory / "test_extra.py").write_text(
        "def test_extra(kernel, cases, run_case):\n"
        "    assert kernel.name == 'scale'\n"
        "    assert [c.name for c in cases] == ['default', 'one_row', 'mine']\n"
        "    assert run_case(cases[1], cases[1].formats[0]).passed\n"
    )
    assert _test(monkeypatch, scale, shim) == 0
    assert "test_extra.py::test_extra PASSED" in capsys.readouterr().out


def test_nightly_cases_run_only_in_the_nightly_target(scale, shim, monkeypatch, capsys):
    toml = scale.directory / "kernel.toml"
    toml.write_text(toml.read_text() + '\n[cases.big]\nrows = 64\ns = 2.0\ntier = "nightly"\n')
    assert _test(monkeypatch, scale, shim) == 0
    assert "test_case[big-" not in capsys.readouterr().out
    assert _test(monkeypatch, scale, shim, "nightly") == 0
    out = capsys.readouterr().out
    assert "test_case[big-fp32] PASSED" in out and "test_case[default-" not in out


def test_bazel_run_runs_the_cli(scale, shim, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))
    toml = str(scale.directory / "kernel.toml")
    assert kernel_test.main([toml, shim, "default", "--case", "one_row", "--format", "e4"]) == 0
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("scale/one_row/e4: PASS  cycles=") and "MAC util=" in line
