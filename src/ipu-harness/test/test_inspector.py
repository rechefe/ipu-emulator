"""The Cycle Inspector's recorder (E07-26): ``--trace`` with ``--context``."""

from __future__ import annotations

import json

import pytest
from scale_kernel import write_scale

from ipu_emu.reference import read_trace
from ipu_harness import cli
from ipu_harness.cases import expand
from ipu_harness.inspector import CONTEXT_VERSION, InspectorRecorder
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat as F
from ipu_isa.registers import REGISTER_DEFINITIONS


@pytest.fixture
def scale(tmp_path):
    return write_scale(tmp_path / "kernels", with_reference=False)


def _cli(*args: object) -> int:
    return cli.main([str(a) for a in args])


def _recorded(scale, tmp_path, *extra: object) -> tuple[bytes, dict]:
    trace, context = tmp_path / "t.jsonl", tmp_path / "c.json"
    assert (
        _cli(scale.directory, "--case", "one_row", *extra, "--trace", trace, "--context", context)
        == 0
    )
    return trace.read_bytes(), json.loads(context.read_text())


def test_the_recorded_trace_is_the_plain_runs_trace(scale, tmp_path):
    plain = tmp_path / "plain.jsonl"
    assert _cli(scale.directory, "--case", "one_row", "--trace", plain) == 0
    recorded, _ = _recorded(scale, tmp_path)
    assert recorded == plain.read_bytes()


def test_the_context_describes_the_program_and_each_word(scale, tmp_path):
    _, context = _recorded(scale, tmp_path, "--format", "e4")
    _, records = read_trace(tmp_path / "t.jsonl")
    assert context["context_version"] == CONTEXT_VERSION
    assert (context["kernel"], context["case"], context["options"]) == ("scale", "one_row", {})
    assert context["source_path"] == "kernels/elementwise/scale/scale.asm"
    assert context["source"] == scale.asm.read_text()
    assert context["labels"] == {"row_loop": 1}
    assert len(context["program"]) == 7
    assert set(context["init"]) == set(REGISTER_DEFINITIONS)
    cr = bytes.fromhex(context["init"]["cr"])
    assert len(cr) == 16 * 4 and int.from_bytes(cr[4:8], "little") == 1  # CR1, the constant 1
    assert len(context["cycles"]) == len(records)
    # Word 1 (LD.MULT of x's one row, row 0 in NX8) reads 128 bytes; word 2 multiplies 128 lanes.
    by_pc = {r.pc_before: facts for r, facts in zip(records, context["cycles"], strict=True)}
    assert by_pc[1] == {"lanes": 0, "xmem_reads": [[0, 128]]}
    assert by_pc[2] == {"lanes": 128, "xmem_reads": []}


def test_a_recorder_records_one_run(scale, tmp_path):
    case = next(c for c in expand(scale) if c.name == "one_row")
    recorder = InspectorRecorder(tmp_path / "t.jsonl")
    with pytest.raises(RuntimeError, match="ran no program"):
        recorder.context(scale, case)
    assert run_kernel(scale, case, F.FP32, RunOptions(executor=recorder)).passed
    with pytest.raises(RuntimeError, match="one run"):
        run_kernel(scale, case, F.FP32, RunOptions(executor=recorder))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--context", "c.json"], "--context goes with --trace"),
        (["--trace", "t.jsonl", "--context", "c.json", "--debug"], "takes no --debug"),
        (["--benchmark", "--context", "c.json"], "takes no --context"),
    ],
)
def test_context_needs_a_trace_and_a_plain_run(scale, tmp_path, monkeypatch, capsys, args, message):
    monkeypatch.chdir(tmp_path)
    assert _cli(scale.directory, *args) == 1
    assert message in capsys.readouterr().err
