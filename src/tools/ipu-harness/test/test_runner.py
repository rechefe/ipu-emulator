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

from ipu_emu.emulator import load_program
from ipu_emu.execute import decode_instruction_word
from ipu_emu.ipu_state import IpuState
from ipu_emu.trace import read_trace, trace_header
from ipu_harness import cli, runner
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.params import plan_case
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


def test_stage_is_the_run_up_to_its_first_word(scale):
    """``stage`` (E07-01's compare_setup reads it) leaves the model halted at PC 0
    with the inputs packed and the registers written; the run continues from it."""
    case = _case(scale, "default")
    staged = runner.stage(scale, case, F.FP32)
    assert staged.model.status.pc == 0
    for write in staged.writes:
        bank, index = write.register[:2].lower(), int(write.register[2:])
        assert staged.model.read_register(bank, index) == write.value
    start, size = staged.region_bytes("x")
    assert (start, size) == (0, 4 * 512)
    x = np.frombuffer(staged.model.read_xmem(start, size), dtype="<f4").reshape(4, 128)
    np.testing.assert_array_equal(x, staged.inputs["x"])
    run = staged.model.run()
    assert run.cycles == run_kernel(scale, case, F.FP32).cycles


def test_a_scratch_region_starts_with_its_fill(tmp_path):
    """Placement only reserves a scratch region; the runner writes its fill."""
    directory = write_scale(tmp_path).directory
    toml = (directory / "kernel.toml").read_text()
    toml = toml.replace(
        "[abi]\n", '[scratch.t]\nshape = ["rows", 100]\nfill = 3\n\n[abi]\nt    = "CR6"\n'
    )
    (directory / "kernel.toml").write_text(toml)
    asm = (directory / "scale.asm").read_text().replace("{{ cr.x }}", "{{ cr.t }}")
    (directory / "scale.asm").write_text(asm)
    manifest = load_manifest(directory)
    result = run_kernel(manifest, _case(manifest, "default"), F.FP32)  # y = t * 2, not x * 2
    # t is 100 wide: its rows' padding lanes hold the fill too, not zeros.
    np.testing.assert_array_equal(result.outputs["y"], np.full((4, 128), 6.0))


def test_the_cycle_limit_is_a_fault_naming_its_word(scale):
    result = run_kernel(scale, _case(scale, "default"), F.INT8, RunOptions(max_cycles=3))
    assert not result.passed
    assert result.fault is not None and result.fault.fault is FaultId.F9
    assert result.comparisons == ()
    assert result.describe().startswith("scale/default/int8: FAIL")
    # Words 0-2 retire (the SETs, LD.MULT, MULT.VE + ACC.ADD.FIRST); word 3 hits the limit.
    assert result.fault.pc == 3 and result.fault.line == "ST.ACC LR2 CR3 ;;"
    assert "fault F-9 at PC 3: ST.ACC LR2 CR3 ;;" in result.describe()


@pytest.mark.parametrize(
    ("rendered", "pc", "line"),
    [
        ("loop:\n    SET LR0 CR0 ;;\n    HALT ;;\n", 0, "SET LR0 CR0 ;;"),
        ("// setup ;; done\nSET LR0 CR0 ;;\nHALT ;;", 0, "SET LR0 CR0 ;;"),
        ("SET LR0 CR0 ; # a ;; in a comment\nSET LR1 CR0 ;;\nHALT ;;", 1, "HALT ;;"),
        (
            "MULT.VE LR0 CR5 0 LR0 CR15 ;\n    ACC.ADD.FIRST ;;\nHALT",
            0,
            "MULT.VE LR0 CR5 0 LR0 CR15 ;",
        ),
        ("SET LR0 CR0 ;;\nHALT\n", 1, "HALT"),  # the last word may omit ";;"
        ("SET LR0 CR0 ;; HALT ;;", 1, "SET LR0 CR0 ;; HALT ;;"),
        ("SET LR0 CR0 ;;\n", 1, None),
    ],
)
def test_the_source_line_of_a_word(rendered, pc, line):
    assert runner._source_line(rendered, pc) == line


def test_the_trace_follows_the_schema(scale, tmp_path):
    trace = tmp_path / "run.jsonl"
    result = run_kernel(scale, _case(scale, "one_row"), F.E4, RunOptions(trace=trace))
    header = json.loads(trace.read_text().splitlines()[0])
    jsonschema.validate(header, load_schema("trace"))
    assert header["format"] == "e4"
    # The header the emulator writes for the same program.
    state = IpuState(fmt=F.E4)
    plan = plan_case(scale, _case(scale, "one_row").values, F.E4)
    load_program(state, [decode_instruction_word(w) for w in assemble(plan).words])
    assert header == trace_header(state)
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


def test_an_input_overrides_the_generator(scale, tmp_path, capsys):
    x = tmp_path / "x.npy"
    np.save(x, np.full((4, 128), 3.0))
    out = tmp_path / "out"
    assert _cli(scale.directory, "--input", f"x={x}", "--output", out, "--format", "fp32") == 0
    np.testing.assert_array_equal(np.load(out / "y.npy"), np.full((4, 128), 6.0))


def test_a_missing_input_file_exits_1(scale, tmp_path, capsys):
    assert _cli(scale.directory, "--input", f"x={tmp_path / 'nope.npy'}") == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


def test_an_assembly_error_in_the_kernel_exits_1(tmp_path, capsys):
    manifest = write_scale(tmp_path, asm="    SET LR0 {{ cr.nope }} ;;\n    HALT ;;\n")
    assert _cli(manifest.directory) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "nope" in err and "Traceback" not in err


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
