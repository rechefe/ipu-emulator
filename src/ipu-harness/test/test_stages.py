"""Multi-stage kernels (E06-14): a toy kernel whose stage 1 copies ``x`` into a
scratch region and whose stage 2 scales that region into ``y``, on one model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ipu_as.errors import TemplateError
from ipu_emu.emulator import DebugAction
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness.abi import register_writes
from ipu_harness.cases import expand
from ipu_harness.debug import DebugHook, DebugLaunch
from ipu_harness.kernel_bin import default_programs, default_stage_programs
from ipu_harness.manifest import load_manifest
from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.placement import place
from ipu_harness.runner import RunOptions, run_kernel, stage
from ipu_harness.stages import StageOptions, run_stages, stage_programs
from ipu_harness.variants import variant
from ipu_isa.cr_halves import ds
from ipu_isa.formats import ElementFormat

E4 = variant("e4")

#: Stage 1: t = x, one row per iteration (an identity multiply, quantized back
#: to the case's NX8 type and stored; the QF half is in {{ cr.qf }}.L).
COPY = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 CR1 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.t }} LR1 ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

#: Stage 2: y = t * s, quantized and stored (the scale kernel's program
#: reading ``t``).
SCALE = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.t }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15.L {{ cr.qf }}.L ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15.L {{ cr.qf }}.L {{ cr.y }} LR2 ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

HOOK = """\
def run(ctx):
    ctx.run_stage("copy.asm", abi={"x": "CR2", "t": "CR3", "rows": "CR4", "qf": "CR14"})
    ctx.run_stage(
        "copy_scale.asm", abi={"t": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5", "qf": "CR14"}
    )
"""

TOML = """\
[kernel]
op = "copy_scale"
formats = ["e4"]
stages = "stages.py:run"

[params]
rows = { type = "int", min = 1, max = 8 }
s    = { type = "float" }

[inputs.x]
shape = ["rows", 128]

[scratch.t]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[derived]
qf = "case_qf"

[abi]
x    = "CR2"
t    = "CR3"
y    = "CR6"
rows = "CR4"
s    = "CR5"
qf   = "CR14"

[golden]
tolerance = { e4 = "EXACT" }

[cases.default]
rows = 3
s = 2.0
inputs = { x = "data/x.npy" }
expected = { y = "data/y.npy" }
"""


def _write(root: Path, hook: str = HOOK) -> Path:
    directory = root / "elementwise" / "copy_scale"
    (directory / "data").mkdir(parents=True)
    x = np.arange(3 * 128, dtype=np.float64).reshape(3, 128) % 17 - 8
    np.save(directory / "data" / "x.npy", x)
    np.save(directory / "data" / "y.npy", x * 2.0)
    (directory / "copy.asm").write_text(COPY)
    (directory / "copy_scale.asm").write_text(SCALE)
    (directory / "stages.py").write_text(hook)
    (directory / "kernel.toml").write_text(TOML)
    return directory


@pytest.fixture
def kernel(tmp_path):
    return load_manifest(_write(tmp_path))


def _default(manifest):
    return next(c for c in expand(manifest) if c.name == "default")


def _alone(manifest, program, abi):
    """The cycles ``program`` takes run as a kernel's only stage."""
    directory = manifest.directory
    (directory / "stages.py").write_text(
        f"def run(ctx):\n    ctx.run_stage({program!r}, abi={abi!r})\n"
    )
    alone = load_manifest(directory)
    return run_kernel(alone, _default(alone), E4).cycles


COPY_ABI = {"x": "CR2", "t": "CR3", "rows": "CR4", "qf": "CR14"}
SCALE_ABI = {"t": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5", "qf": "CR14"}


def test_two_stages_on_one_model_give_the_expected_output(tmp_path):
    kernel = load_manifest(_write(tmp_path))
    result = run_kernel(kernel, _default(kernel), E4)
    assert result.passed, result.describe()
    assert [program for program, _ in result.stages] == ["copy.asm", "copy_scale.asm"]
    # Each stage's count is that program's own, and the run's is their sum.
    assert dict(result.stages) == {
        "copy.asm": _alone(kernel, "copy.asm", COPY_ABI),
        "copy_scale.asm": _alone(kernel, "copy_scale.asm", SCALE_ABI),
    }
    assert result.cycles == sum(cycles for _, cycles in result.stages)
    assert "stages copy.asm=" in result.describe()


def test_a_stage_without_its_own_abi_gets_the_kernels_offsets(tmp_path, monkeypatch):
    from ipu_harness import stages

    directory = _write(tmp_path, 'def run(ctx):\n    ctx.run_stage("copy.asm")\n')
    toml = directory / "kernel.toml"
    toml.write_text(
        toml.read_text().replace('y    = "CR6"', 'y    = { register = "CR6", offset = -1 }')
    )
    seen = []

    def recording(*args, **kwargs):
        writes = register_writes(*args, **kwargs)
        seen.append({w.register: w.value for w in writes})
        return writes

    monkeypatch.setattr(stages, "register_writes", recording)
    manifest = load_manifest(directory)
    result = run_kernel(manifest, _default(manifest), E4)
    assert len(seen) == 1  # the one stage
    assert seen[0]["CR6"] == result.placement.region("y").base_row - 1


def test_a_kernel_whose_files_are_symlinks_runs_its_stages(tmp_path):
    """Bazel's runfiles: real folders, files linked to the checkout."""
    source = _write(tmp_path / "src")
    linked = tmp_path / "runfiles" / "elementwise" / "copy_scale"
    (linked / "data").mkdir(parents=True)
    for file in source.rglob("*"):
        if file.is_file():
            (linked / file.relative_to(source)).symlink_to(file)
    manifest = load_manifest(linked)
    assert run_kernel(manifest, _default(manifest), E4).passed


def test_run_stages_stops_at_every_stage_entry_under_a_debugger(kernel):
    plan = plan_case(kernel, _default(kernel).values, E4)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in kernel.tensors.items()})
    stops = []

    def debugger(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    staged = run_stages(
        plan,
        placement,
        ReferenceModel(ElementFormat.NX8),
        options=StageOptions(debug=DebugHook(debugger), break_on_entry=True),
    )
    assert stops == [0, 0]  # each stage's entry
    assert [s.program for s in staged.stages] == ["copy.asm", "copy_scale.asm"]


def test_each_stage_writes_its_own_trace(kernel, tmp_path):
    trace = tmp_path / "out" / "run.jsonl"
    trace.parent.mkdir()
    result = run_kernel(kernel, _default(kernel), E4, RunOptions(trace=trace))
    assert result.passed
    headers = [
        json.loads(trace.with_name(f"run.{i}.jsonl").read_text().splitlines()[0]) for i in (0, 1)
    ]
    assert headers[0]["program_digest"] != headers[1]["program_digest"]
    assert not trace.exists()


def test_a_stage_gets_its_params_and_ds(tmp_path):
    hook = (
        "from ipu_isa.cr_halves import ds\n"
        "def run(ctx):\n"
        f"    ctx.run_stage('copy.asm', abi={COPY_ABI!r})\n"
        f"    ctx.run_stage('param.asm', abi={SCALE_ABI!r}, params={{'k': 2}},\n"
        "                  ds=ds(64))\n"
    )
    directory = _write(tmp_path, hook)
    # The stage's own value k picks the scale register: s, bound to CR5.
    (directory / "param.asm").write_text(SCALE.replace("{{ cr.s }}", "CR{{ k + 3 }}"))
    manifest = load_manifest(directory)
    plan = plan_case(manifest, _default(manifest).values, E4)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in manifest.tensors.items()})
    model = ReferenceModel(ElementFormat.NX8)
    staged = run_stages(plan, placement, model)
    assert [s.program for s in staged.stages] == ["copy.asm", "param.asm"]
    assert model.read_register("cr", 15) == ds(64)  # CR15.L, the stage's DS


@pytest.mark.parametrize(
    ("call", "match"),
    [
        ("ctx.run_stage('copy.asm', abi={'x': 'CR2', 't': 'CR2', 'rows': 'CR4'})", "both bind CR2"),
        ("ctx.run_stage('copy.asm', abi={'x': 'CR2', 't': 'CR3', 'rows': 'CR0'})", "hardwired"),
        ("ctx.run_stage('copy.asm', abi={'x': 'CR2', 't': 'CR3', 'rows': 'CR99'})", "no CR or LR"),
        ("ctx.run_stage('copy_scale.asm', abi={'s': 'LR5'})", "binds a CR, not LR5"),
        ("ctx.run_stage('copy.asm', abi={'zz': 'CR2'})", "'zz' names no parameter"),
        ("ctx.run_stage('copy.asm', params={'rows': 1})", "already the case's"),
        (
            "ctx.run_stage('copy.asm', abi={'x': 'CR2', 't': 'CR3', 'rows': 'CR15',"
            " 'qf': 'CR14'}, ds=ds(128))",
            "binds CR15 and a DS",
        ),
        ("pass", "ran no stage"),
    ],
)
def test_a_wrong_stage_request_fails_before_it_runs(tmp_path, call, match):
    hook = f"from ipu_isa.cr_halves import ds\ndef run(ctx):\n    {call}\n"
    manifest = load_manifest(_write(tmp_path, hook))
    with pytest.raises(ConfigurationError, match=match):
        run_kernel(manifest, _default(manifest), E4)


def test_a_stage_outside_the_family_fails_before_it_runs(tmp_path):
    hook = 'def run(ctx):\n    ctx.run_stage("../../other/k/k.asm")\n'
    manifest = load_manifest(_write(tmp_path, hook))
    with pytest.raises(ConfigurationError, match="outside the kernel's family"):
        run_kernel(manifest, _default(manifest), E4)


def test_a_stage_may_run_a_sibling_kernels_program(tmp_path):
    directory = _write(
        tmp_path,
        f'def run(ctx):\n    ctx.run_stage("../sibling/copy.asm", abi={COPY_ABI!r})\n'
        f'    ctx.run_stage("copy_scale.asm", abi={SCALE_ABI!r})\n',
    )
    sibling = directory.parent / "sibling"
    sibling.mkdir()
    (sibling / "copy.asm").write_text(COPY)
    manifest = load_manifest(directory)
    assert run_kernel(manifest, _default(manifest), E4).passed


def test_a_stage_naming_a_register_its_abi_does_not_bind_fails(tmp_path):
    hook = 'def run(ctx):\n    ctx.run_stage("copy.asm", abi={"x": "CR2", "rows": "CR4", "qf": "CR14"})\n'
    manifest = load_manifest(_write(tmp_path, hook))
    with pytest.raises(TemplateError, match="cr.t"):
        run_kernel(manifest, _default(manifest), E4)


def test_a_stage_reading_a_cr_it_does_not_bind_fails(tmp_path):
    directory = _write(tmp_path, 'def run(ctx):\n    ctx.run_stage("raw.asm", abi={})\n')
    (directory / "raw.asm").write_text("SET LR0 CR7 ;;\nHALT ;;\n")
    manifest = load_manifest(directory)
    with pytest.raises(ConfigurationError, match="CR7"):
        run_kernel(manifest, _default(manifest), E4)


CAUGHT = "    except {kind}:\n        open(__file__ + '.caught', 'w').close()\n"


@pytest.mark.parametrize("catching", [None, "Exception", "BaseException"])
def test_a_stage_that_faults_ends_the_run_with_its_fault(tmp_path, catching):
    """Even when the hook catches what a stage raises: an ``except Exception``
    never sees the stop, and a stage after a caught stop stops again."""
    first = f"    ctx.run_stage('copy.asm', abi={COPY_ABI!r})\n"
    if catching is not None:
        first = "    try:\n    " + first + CAUGHT.format(kind=catching)
    body = first + f"    ctx.run_stage('copy_scale.asm', abi={SCALE_ABI!r})\n"
    directory = _write(tmp_path, "def run(ctx):\n" + body)
    manifest = load_manifest(directory)
    result = run_kernel(manifest, _default(manifest), E4, RunOptions(max_cycles=5))
    assert result.fault is not None and result.fault.fault.label == "F-9"
    assert [program for program, _ in result.stages] == ["copy.asm"]
    # Only a BaseException handler sees the stop (and the second stage stops again).
    assert (directory / "stages.py.caught").exists() == (catching == "BaseException")


def _debug(callback):
    return DebugLaunch(break_on_entry=True, factory=lambda kernel: DebugHook(callback))


def test_a_debug_run_stops_in_every_stage(kernel):
    stops = []

    def debugger(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    result = run_kernel(kernel, _default(kernel), E4, RunOptions(debug=_debug(debugger)))
    assert stops == [0, 0] and result.passed and len(result.stages) == 2


def test_a_quit_in_a_stage_ends_the_run_without_the_rest(kernel):
    result = run_kernel(
        kernel,
        _default(kernel),
        E4,
        RunOptions(debug=_debug(lambda state, cycles: DebugAction.QUIT)),
    )
    assert result.quit and not result.passed
    assert [program for program, _ in result.stages] == ["copy.asm"]


def test_one_alias_profile_covers_every_stage(kernel):
    result = run_kernel(kernel, _default(kernel), E4, RunOptions(alias_profile=True))
    assert result.passed and result.alias_profile is not None
    assert result.alias_profile.cycles == result.cycles


def test_a_multi_stage_kernel_has_no_one_program_to_stage_or_execute(kernel):
    with pytest.raises(ConfigurationError, match="no one program"):
        stage(kernel, _default(kernel), E4)
    options = RunOptions(executor=lambda context: None)
    with pytest.raises(ConfigurationError, match="executor runs one program"):
        run_kernel(kernel, _default(kernel), E4, options)


def test_stage_programs_assembles_every_stage_and_runs_none(kernel):
    plan = plan_case(kernel, _default(kernel).values, E4)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in kernel.tensors.items()})
    programs = stage_programs(plan, placement)
    assert [name for name, _ in programs] == ["copy.asm", "copy_scale.asm"]
    assert programs[0][1].words != programs[1][1].words
    assert all(len(assembled.words) > 0 for _, assembled in programs)


def test_the_assemble_target_of_a_multi_stage_kernel_is_its_own_program(kernel):
    # :assemble_<name> (kernel_bin) writes the stage that runs <name>.asm.
    own = dict(stage_programs_of(kernel))["copy_scale.asm"]
    assert default_programs(kernel)[E4].words == own.words
    assert [name for name, _ in default_stage_programs(kernel)[E4]] == [
        "copy.asm",
        "copy_scale.asm",
    ]


def stage_programs_of(manifest):
    plan = plan_case(manifest, _default(manifest).values, E4)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in manifest.tensors.items()})
    return stage_programs(plan, placement)


def test_a_multi_stage_run_sums_its_stages_statistics(kernel):
    # The benchmark's use columns (E06-11) for a multi-stage kernel: each
    # stage's counts summed, the ratios over the sum (E07-19).
    result = run_kernel(kernel, _default(kernel), E4)
    stats = result.stats
    assert stats is not None and result.passed
    assert stats.cycles == result.cycles == sum(cycles for _, cycles in result.stages)
    assert stats.mult_active_cycles > 0 and stats.xmem_writes > 0
    assert stats.lane_occupancy == stats.mult_lane_ops / (stats.cycles * 128)
    assert stats.effective_mac_utilization == result.mac_utilization
