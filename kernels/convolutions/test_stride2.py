"""The stride-2 depthwise kernels' two stages (E07-19): each runs
``depthwise_conv_tpl``'s program at full resolution, then its own decimating
program (``stages.py``). Beyond each kernel's cases: the stages add up to the
run, a debugger stops in stage 2 too, and stage 1's values (a copy of the
template's ``[derived]``, since a kernel cannot share a sibling's) stay the
template's.
"""

from __future__ import annotations

import pytest

from ipu_emu.emulator import DebugAction
from ipu_harness.cases import expand
from ipu_harness.debug import DebugHook, DebugLaunch
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat

STRIDE2 = [
    "depthwise_conv_stride2_narrow",
    "depthwise_conv_stride2_128",
    "depthwise_conv_stride2_16",
]
STAGE1 = "../depthwise_conv_tpl/depthwise_conv_tpl.asm"


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def _default(kernel):
    return next(case for case in expand(kernel) if case.name == "default")


@pytest.mark.parametrize("name", STRIDE2)
def test_stages_add_up(registry, name):
    kernel = registry.kernel(name)
    result = run_kernel(kernel, _default(kernel), ElementFormat.FP32)
    assert result.passed, result
    assert [program for program, _ in result.stages] == [STAGE1, f"{name}.asm"]
    assert result.cycles == sum(cycles for _, cycles in result.stages)
    assert 0 < result.mac_utilization <= 1.0


@pytest.mark.parametrize("name", STRIDE2)
def test_debug_stops_in_stage_2(registry, name):
    # The old harness gave stage 2 no debug callback; a stage now gets the
    # run's hook, so a debug launch stops on entry to each stage.
    kernel = registry.kernel(name)
    stops = []

    def debugger(state, cycles):
        stops.append(state.program_counter)
        return DebugAction.CONTINUE

    launch = DebugLaunch(break_on_entry=True, factory=lambda kernel: DebugHook(debugger))
    result = run_kernel(kernel, _default(kernel), ElementFormat.FP32, RunOptions(debug=launch))
    assert result.passed, result
    assert stops == [0, 0]
    assert len(result.stages) == 2


@pytest.mark.parametrize("name", STRIDE2)
def test_stage_1_reads_the_templates_values(registry, name):
    # Every non-tensor value stage 1's program reads ([abi] of
    # depthwise_conv_tpl) is the template's, at the same shape, stride 1.
    kernel, template = registry.kernel(name), registry.kernel("depthwise_conv_tpl")
    case = _default(kernel)
    values = {k: v for k, v in case.values.items() if k in template.params}
    plan = plan_case(kernel, case.values, ElementFormat.FP32, case=case.decl)
    plain = plan_case(template, values | dict(stride=1, fpb=28), ElementFormat.FP32)
    names = [n for n in template.abi if n not in template.tensors]
    # kernel_rows (CR8, written, never read) is where the final output is parked.
    assert [n for n in names if n not in kernel.abi] == ["kernel_rows"]
    assert kernel.abi["out"] == template.abi["kernel_rows"]
    for n in (n for n in names if n in kernel.abi):
        assert kernel.abi[n] == template.abi[n], n
        if n in kernel.derived and n not in plan.derived:
            # Known only when the registers are written (it reads case_qf):
            # the same expression.
            assert kernel.derived[n] == template.derived[n], n
            continue
        mine = plan.derived[n] if n in plan.derived else plan.params[n]
        theirs = plain.derived[n] if n in plain.derived else plain.params[n]
        assert mine == theirs, n
