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
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import IpuState
from ipu_emu.trace import read_trace, trace_header
from ipu_harness import cli, runner
from ipu_harness.assembly import assemble
from ipu_harness.cases import expand
from ipu_harness.manifest import load_manifest
from ipu_harness.params import plan_case
from ipu_harness.runner import RunOptions, run_kernel
from ipu_harness.variants import variant
from ipu_isa.codec.word import DecodedWord
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat
from ipu_isa.trace_schema import load_schema

FP32, E4 = variant("fp32"), variant("e4")


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
    assert runs == 6  # 3 cases x 2 variants


def test_the_report_line(scale):
    result = run_kernel(scale, _case(scale, "one_row"), E4)
    assert re.fullmatch(
        r"scale/one_row/e4: PASS  cycles=\d+  MAC util=\d+\.\d%  y: max\|err\|=0 \(EXACT\)",
        result.describe(),
    )


def test_raw_outputs_are_the_region_bytes(scale):
    result = run_kernel(scale, _case(scale, "one_row"), FP32)
    raw = result.raw_outputs["y"]
    assert len(raw) == 128 * 4  # one FP32 row
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<f4").reshape(1, 128), result.outputs["y"]
    )


def test_stage_is_the_run_up_to_its_first_word(scale):
    """``stage`` leaves the model halted at PC 0 with the inputs packed and the
    registers written; the run continues from it."""
    case = _case(scale, "default")
    staged = runner.stage(scale, case, FP32)
    assert staged.model.status.pc == 0
    for write in staged.writes:
        bank, index = write.register[:2].lower(), int(write.register[2:])
        assert staged.model.read_register(bank, index) == write.value
    start, size = staged.region_bytes("x")
    assert (start, size) == (0, 4 * 512)
    x = np.frombuffer(staged.model.read_xmem(start, size), dtype="<f4").reshape(4, 128)
    np.testing.assert_array_equal(x, staged.inputs["x"])
    run = staged.model.run()
    assert run.cycles == run_kernel(scale, case, FP32).cycles


def test_a_scratch_region_starts_with_its_fill(tmp_path):
    """Placement only reserves a scratch region; the runner writes its fill."""
    directory = write_scale(tmp_path).directory
    toml = (directory / "kernel.toml").read_text()
    toml = toml.replace(
        "[abi]\n", '[scratch.t]\nshape = ["rows", 100]\nfill = 3\n\n[abi]\nt    = "CR7"\n'
    )
    (directory / "kernel.toml").write_text(toml)
    asm = (directory / "scale.asm").read_text().replace("{{ cr.x }}", "{{ cr.t }}")
    (directory / "scale.asm").write_text(asm)
    manifest = load_manifest(directory)
    result = run_kernel(manifest, _case(manifest, "default"), FP32)  # y = t * 2, not x * 2
    # t is 100 wide: its rows' padding lanes hold the fill too, not zeros.
    np.testing.assert_array_equal(result.outputs["y"], np.full((4, 128), 6.0))


def test_a_bytes_region_starts_with_its_functions_bytes(tmp_path):
    directory = write_scale(tmp_path).directory
    (directory / "mask.py").write_text(
        "def mask(plan):\n    return bytes(range(256)) * 2  # one FP32 row\n"
    )
    toml = (directory / "kernel.toml").read_text()
    toml = toml.replace(
        "[abi]\n", '[scratch.m]\nshape = [1, 128]\nbytes = "mask.py:mask"\n\n[abi]\nm    = "CR7"\n'
    )
    (directory / "kernel.toml").write_text(toml)
    manifest = load_manifest(directory)
    staged = runner.stage(manifest, _case(manifest, "default"), FP32)
    assert staged.model.read_xmem(*staged.region_bytes("m")) == bytes(range(256)) * 2


CONDITIONAL_TOML = """\
[kernel]
op = "conv"
formats = ["fp32"]

[params]
rows = { type = "int", min = 1 }
bias = { type = "bool", default = false }

[inputs.x]
shape = ["rows", 128]

[inputs.b]
shape = [1, 128]
when = "bias"

[outputs.y]
shape = ["rows", 128]

[abi]
x = "CR2"
y = "CR3"
b = "CR10"

[cases.default]
rows = 1
bias = [false, true]
"""


def test_a_cr_bound_only_under_another_cases_condition_is_an_unbound_read(tmp_path):
    """CR10 holds the bias base only when ``bias``; a case without it reading
    CR10 reads an unbound register, though the kernel's ``[abi]`` lists it."""
    directory = tmp_path / "conv"
    directory.mkdir()
    (directory / "conv.asm").write_text("ADD LR0 LR0 CR10 ;;\nHALT ;;\n")
    (directory / "kernel.toml").write_text(CONDITIONAL_TOML)
    manifest = load_manifest(directory)
    cases = {case.values["bias"]: case for case in expand(manifest)}
    runner.stage(manifest, cases[True], FP32)
    with pytest.raises(ConfigurationError, match="reads CR10"):
        runner.stage(manifest, cases[False], FP32)


def test_the_cycle_limit_is_a_fault_naming_its_word(scale):
    result = run_kernel(scale, _case(scale, "default"), E4, RunOptions(max_cycles=3))
    assert not result.passed
    assert result.fault is not None and result.fault.fault is FaultId.F9
    assert result.comparisons == ()
    assert result.describe().startswith("scale/default/e4: FAIL")
    # Words 0-2 retire (the SETs, LD.MULT, MULT.VE + ACC.ADD.FIRST); word 3 hits the limit.
    store = "ACTIVATE.QUANTIZE identity CR15.L CR6.L CR3 LR2 ;;"
    assert result.fault.pc == 3 and result.fault.line == store
    assert f"fault F-9 at PC 3: {store}" in result.describe()


@pytest.mark.parametrize(
    ("rendered", "pc", "line"),
    [
        ("loop:\n    SET LR0 CR0 ;;\n    HALT ;;\n", 0, "SET LR0 CR0 ;;"),
        ("// setup ;; done\nSET LR0 CR0 ;;\nHALT ;;", 0, "SET LR0 CR0 ;;"),
        ("SET LR0 CR0 ; # a ;; in a comment\nSET LR1 CR0 ;;\nHALT ;;", 1, "HALT ;;"),
        (
            "MULT.VE LR0 CR5 0 LR0 CR15.L CR14.L ;\n    ACC.ADD.FIRST ;;\nHALT",
            0,
            "MULT.VE LR0 CR5 0 LR0 CR15.L CR14.L ;",
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
    result = run_kernel(scale, _case(scale, "one_row"), E4, RunOptions(trace=trace))
    header = json.loads(trace.read_text().splitlines()[0])
    jsonschema.validate(header, load_schema("trace"))
    assert header["format"] == "nx8"  # the run mode (the variant is the harness's)
    # The header the emulator writes for the same program.
    state = IpuState(fmt=ElementFormat.NX8)
    plan = plan_case(scale, _case(scale, "one_row").values, E4)
    load_program(state, [DecodedWord.from_word(w) for w in assemble(plan).words])
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
        "scale/default/e4",
    ]


def test_the_cli_lists_the_cases(scale, capsys):
    assert _cli(scale.directory, "--list-cases") == 0
    assert capsys.readouterr().out.splitlines() == [
        "default  (fp32, e4)",
        "one_row  (fp32, e4)",
        "mine  (fp32, e4)",
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


def test_activate_exact_is_refused_in_an_nx8_case_and_runs_in_fp32(tmp_path):
    """ADR 0007: ACTIVATE.EXACT stores an FP32 row; an NX8 case refuses the
    program when the model loads it, before anything runs."""
    directory = write_scale(tmp_path).directory
    asm = (directory / "scale.asm").read_text()
    (directory / "scale.asm").write_text(
        asm.replace(
            "ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR2",
            "ACTIVATE.EXACT identity CR15.L {{ cr.y }} LR2",
        )
    )
    manifest = load_manifest(directory)
    case = _case(manifest, "default")
    with pytest.raises(ConfigurationError, match="valid in FP32 mode only"):
        run_kernel(manifest, case, E4)
    assert run_kernel(manifest, case, FP32).passed


@pytest.mark.parametrize("name", ["e2", "e5", "dynamic", "s0e3w6"])
def test_scale_passes_exactly_in_other_nx8_variants(tmp_path, name):
    """Inputs quantized row by row (a dynamic row picks its exponent split; an
    unsigned type stores a negative as 0), the golden quantized the same way."""
    directory = write_scale(tmp_path).directory
    toml = directory / "kernel.toml"
    toml.write_text(
        toml.read_text()
        .replace('formats = ["fp32", "e4"]', f'formats = ["fp32", "e4", "{name}"]')
        .replace('e4 = "EXACT"', f'e4 = "EXACT", {name} = "EXACT"')
    )
    manifest = load_manifest(directory)
    for case in expand(manifest):
        result = run_kernel(manifest, case, variant(name))
        assert result.passed, result.describe()
