"""The runner: one case in one format, from the manifest to a verdict (E06-11).

The epic's ``scale`` kernel (``scale_kernel.py``) runs end to end through
:func:`run_kernel` and through the command line. This target has torch, for
the kernel's reference; ``test_runner_no_torch`` runs it without.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from scale_kernel import write_scale

from ipu_emu.trace import read_trace
from ipu_harness import cli
from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat as F
from ipu_isa.trace_schema import load_schema


@pytest.fixture
def scale(tmp_path):
    return write_scale(tmp_path)


def _case(manifest, name):
    return next(case for case in expand(manifest) if case.name == name)


def test_every_case_and_format_passes(scale):
    runs = 0
    for case in expand(scale):
        for fmt in case.formats:
            result = run_kernel(scale, case, fmt)
            assert result.passed, result.describe()
            assert result.cycles > 0
            assert 0 < result.mac_utilization <= 1
            assert [c.output for c in result.comparisons] == ["y"]
            runs += 1
    assert runs == 9  # 3 cases x 3 formats


def test_the_report_line(scale):
    result = run_kernel(scale, _case(scale, "one_row"), F.E4)
    assert re.fullmatch(
        r"scale/one_row/e4: PASS  cycles=\d+  MAC util=\d+\.\d%  y: max\|err\|=0 \(EXACT\)",
        result.describe(),
    )


def test_raw_outputs_are_the_region_bytes(scale):
    result = run_kernel(scale, _case(scale, "one_row"), F.FP32)
    raw = result.raw_outputs["y"]
    assert len(raw) == 128 * 4  # one FP32 row
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<f4").reshape(1, 128), result.outputs["y"]
    )


def test_a_scratch_region_starts_with_its_fill(tmp_path):
    """Placement only reserves a scratch region; the runner writes its fill."""
    directory = write_scale(tmp_path).directory
    toml = (directory / "kernel.toml").read_text()
    toml = toml.replace(
        "[abi]\n", '[scratch.t]\nshape = ["rows", 128]\nfill = 3\n\n[abi]\nt    = "CR6"\n'
    )
    (directory / "kernel.toml").write_text(toml)
    asm = (directory / "scale.asm").read_text().replace("{{ cr.x }}", "{{ cr.t }}")
    (directory / "scale.asm").write_text(asm)
    manifest = load_manifest(directory)
    result = run_kernel(manifest, _case(manifest, "default"), F.FP32)  # y = t * 2, not x * 2
    np.testing.assert_array_equal(result.outputs["y"], np.full((4, 128), 6.0))


def test_the_cycle_limit_is_a_fault_naming_its_word(scale):
    result = run_kernel(scale, _case(scale, "default"), F.INT8, RunOptions(max_cycles=3))
    assert not result.passed
    assert result.fault is not None and result.fault.fault is FaultId.F9
    assert result.comparisons == ()
    assert result.describe().startswith("scale/default/int8: FAIL")
    assert f"fault F-9 at PC {result.fault.pc}" in result.describe()
    assert result.fault.line is not None and result.fault.line.endswith(";;")


def test_the_trace_follows_the_schema(scale, tmp_path):
    trace = tmp_path / "run.jsonl"
    result = run_kernel(scale, _case(scale, "one_row"), F.E4, RunOptions(trace=trace))
    header = json.loads(trace.read_text().splitlines()[0])
    jsonschema.validate(header, load_schema("trace"))
    assert header["format"] == "e4"
    _, records = read_trace(trace)
    assert len(records) == result.cycles
    record_schema = load_schema("commit_record")
    for record in records:
        jsonschema.validate(record.to_json(), record_schema)
    assert records[-1].outcome == "halt"


# --- the command line ---------------------------------------------------------------------


def _cli(*args: str) -> int:
    return cli.main([str(a) for a in args])


def test_the_cli_runs_a_case_in_every_format(scale, capsys):
    assert _cli(scale.directory, "--case", "default", "--format", "all") == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in lines] == [
        "scale/default/fp32",
        "scale/default/int8",
        "scale/default/e4",
    ]


def test_the_cli_lists_the_cases(scale, capsys):
    assert _cli(scale.directory, "--list-cases") == 0
    assert capsys.readouterr().out.splitlines() == [
        "default  (fp32, int8, e4)",
        "one_row  (fp32, int8, e4)",
        "mine  (fp32, int8, e4)",
    ]


def test_input_overrides_and_outputs_are_written_even_on_failure(scale, tmp_path, capsys):
    x = tmp_path / "x.npy"
    np.save(x, np.ones((2, 128)))
    out = tmp_path / "out"
    # The case's expected y is 2 * data/x.npy, so ones fail it.
    code = _cli(scale.directory, "--case", "mine", "--input", f"x={x}", "--output", out)
    assert code == 1
    assert "FAIL" in capsys.readouterr().out
    np.testing.assert_array_equal(np.load(out / "y.npy"), np.full((2, 128), 2.0))


def test_a_cycle_limit_exits_1_with_the_fault(scale, capsys):
    assert _cli(scale.directory, "--max-cycles", "3") == 1
    assert "fault F-9 at PC" in capsys.readouterr().out


def test_a_bad_request_exits_1_in_one_line(scale, capsys):
    assert _cli(scale.directory, "--case", "nope") == 1
    err = capsys.readouterr().err
    assert err.startswith("error: scale has no case 'nope'") and "Traceback" not in err


def test_a_bug_in_a_generator_exits_2_with_a_traceback(scale, capsys):
    directory: Path = scale.directory
    (directory / "gen.py").write_text("def make(plan, rng):\n    return None.x\n")
    toml = (directory / "kernel.toml").read_text()
    (directory / "kernel.toml").write_text(
        toml + '\n[cases.planted]\nrows = 1\ns = 2.0\ngenerator = "gen.py:make"\n'
    )
    assert _cli(directory, "--case", "planted") == 2
    err = capsys.readouterr().err
    assert "Traceback" in err and "AttributeError" in err
