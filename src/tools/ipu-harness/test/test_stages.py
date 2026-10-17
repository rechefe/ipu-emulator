"""Multi-stage kernels (E06-14): a toy kernel whose stage 1 copies ``x`` into a
scratch region and whose stage 2 scales that region into ``y``, on one model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_as.errors import TemplateError
from ipu_emu.emulator import DebugAction
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness.cases import expand
from ipu_harness.debug import DebugHook, DebugLaunch
from ipu_harness.manifest import load_manifest
from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.placement import place
from ipu_harness.runner import RunOptions, run_kernel
from ipu_harness.stages import StageOptions, run_stages
from ipu_isa.formats import ElementFormat

#: Stage 1: t = x, one row per iteration (an identity multiply, quantized back
#: to INT8 and stored).
COPY = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.x }} ;;
    MULT.VE LR0 CR1 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ACTIVATE.QUANTIZE identity CR15 ;;
    ST.AAQ LR1 {{ cr.t }} ;;
    ADD LR1 LR1 CR1 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

#: Stage 2: y = t * s, the accumulator stored (the scale kernel's program
#: reading ``t``).
SCALE = """\
    SET LR0 CR0 ;
    SET LR1 CR0 ;
    SET LR2 CR0 ;;
row_loop:
    LD.MULT R0 LR1 {{ cr.t }} ;;
    MULT.VE LR0 {{ cr.s }} 0 LR0 CR15 ;
    ACC.ADD.FIRST ;;
    ST.ACC LR2 {{ cr.y }} ;;
    ADD LR1 LR1 CR1 ;
    INC LR2 4 ;;
    BLT.S LR1 {{ cr.rows }} row_loop ;;
    HALT ;;
"""

HOOK = """\
def run(ctx):
    ctx.run_stage("copy.asm", abi={"x": "CR2", "t": "CR3", "rows": "CR4"})
    ctx.run_stage("copy_scale.asm", abi={"t": "CR2", "y": "CR3", "rows": "CR4", "s": "CR5"})
"""

TOML = """\
[kernel]
op = "copy_scale"
formats = ["int8"]
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
store = "accumulator"

[abi]
x    = "CR2"
t    = "CR3"
y    = "CR6"
rows = "CR4"
s    = "CR5"

[golden]
tolerance = { int8 = "EXACT" }

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


def test_two_stages_on_one_model_give_the_reference(kernel):
    result = run_kernel(kernel, _default(kernel), ElementFormat.INT8)
    assert result.passed, result.describe()
    assert [program for program, _ in result.stages] == ["copy.asm", "copy_scale.asm"]
    assert result.cycles == sum(cycles for _, cycles in result.stages)
    assert all(cycles > 0 for _, cycles in result.stages)
    assert "stages copy.asm=" in result.describe()


def test_every_stage_stops_at_entry_under_a_debugger(kernel):
    plan = plan_case(kernel, _default(kernel).values, ElementFormat.INT8)
    placement = place(plan, {n: stored_rows(t, plan) for n, t in kernel.tensors.items()})
    model = ReferenceModel(ElementFormat.INT8)
    stops = []

    def debugger(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    staged = run_stages(
        plan, placement, model, options=StageOptions(debug=DebugHook(debugger), break_on_entry=True)
    )
    assert stops == [0, 0]  # each stage's entry
    assert len(staged.stages) == 2 and staged.cycles == sum(s.cycles for s in staged.stages)


def test_a_stage_outside_the_family_fails_before_it_runs(tmp_path):
    hook = 'def run(ctx):\n    ctx.run_stage("../../other/k/k.asm")\n'
    manifest = load_manifest(_write(tmp_path, hook))
    with pytest.raises(ConfigurationError, match="outside the kernel's family"):
        run_kernel(manifest, _default(manifest), ElementFormat.INT8)


def test_a_stage_may_run_a_sibling_kernels_program(tmp_path):
    directory = _write(
        tmp_path,
        'def run(ctx):\n    ctx.run_stage("../sibling/copy.asm", abi={"x": "CR2", "t": "CR3",'
        ' "rows": "CR4"})\n    ctx.run_stage("copy_scale.asm", abi={"t": "CR2", "y": "CR3",'
        ' "rows": "CR4", "s": "CR5"})\n',
    )
    sibling = directory.parent / "sibling"
    sibling.mkdir()
    (sibling / "copy.asm").write_text(COPY)
    manifest = load_manifest(directory)
    assert run_kernel(manifest, _default(manifest), ElementFormat.INT8).passed


def test_a_stage_naming_a_register_its_abi_does_not_bind_fails(tmp_path):
    hook = 'def run(ctx):\n    ctx.run_stage("copy.asm", abi={"x": "CR2", "rows": "CR4"})\n'
    manifest = load_manifest(_write(tmp_path, hook))
    with pytest.raises(TemplateError, match="cr.t"):
        run_kernel(manifest, _default(manifest), ElementFormat.INT8)


def test_a_stage_reading_a_cr_it_does_not_bind_fails(tmp_path):
    directory = _write(tmp_path, 'def run(ctx):\n    ctx.run_stage("raw.asm", abi={})\n')
    (directory / "raw.asm").write_text("SET LR0 CR7 ;;\nHALT ;;\n")
    manifest = load_manifest(directory)
    with pytest.raises(ConfigurationError, match="CR7"):
        run_kernel(manifest, _default(manifest), ElementFormat.INT8)


def test_a_stage_that_faults_ends_the_run_with_its_fault(kernel):
    result = run_kernel(kernel, _default(kernel), ElementFormat.INT8, RunOptions(max_cycles=5))
    assert result.fault is not None and result.fault.fault.label == "F-9"
    assert [program for program, _ in result.stages] == ["copy.asm"]


def _debug(callback):
    return DebugLaunch(break_on_entry=True, factory=lambda kernel: DebugHook(callback))


def test_a_debug_run_stops_in_every_stage(kernel):
    stops = []

    def debugger(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    result = run_kernel(
        kernel, _default(kernel), ElementFormat.INT8, RunOptions(debug=_debug(debugger))
    )
    assert stops == [0, 0] and result.passed and len(result.stages) == 2


def test_a_quit_in_a_stage_ends_the_run_without_the_rest(kernel):
    result = run_kernel(
        kernel,
        _default(kernel),
        ElementFormat.INT8,
        RunOptions(debug=_debug(lambda state, cycles: DebugAction.QUIT)),
    )
    assert result.quit and not result.passed
    assert [program for program, _ in result.stages] == ["copy.asm"]


def test_one_alias_profile_covers_every_stage(kernel):
    result = run_kernel(
        kernel, _default(kernel), ElementFormat.INT8, RunOptions(alias_profile=True)
    )
    assert result.passed and result.alias_profile is not None
    assert result.alias_profile.cycles == result.cycles
