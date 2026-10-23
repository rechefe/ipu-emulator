"""The runner without torch (E06-11): a case with expected files needs no reference.

This target has no torch dependency, so an import of torch would fail; the
test also checks torch is not loaded. A spy shows each run plans its case
exactly once (E06-03).
"""

from __future__ import annotations

import sys

import pytest
from scale_kernel import write_scale

import ipu_harness
from ipu_emu.errors import ConfigurationError
from ipu_harness import params
from ipu_harness.cases import expand
from ipu_harness.debug import DebugLaunch
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.faults import FaultId
from ipu_isa.formats import ElementFormat


def test_expected_files_need_no_torch(tmp_path):
    manifest = write_scale(tmp_path, with_reference=False)
    for case in expand(manifest):
        for fmt in case.formats:
            result = run_kernel(manifest, case, fmt)
            assert result.passed, result.describe()
    assert "torch" not in sys.modules


def test_each_run_plans_its_case_once(tmp_path, monkeypatch):
    manifest = write_scale(tmp_path, with_reference=False)
    calls = []
    real = params.plan_case

    def spy(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)

    # Every module of the package that holds plan_case under that name.
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith(ipu_harness.__name__) and getattr(module, "plan_case", None) is real:
            monkeypatch.setattr(module, "plan_case", spy)
    case = next(c for c in expand(manifest) if c.name == "default")
    run_kernel(manifest, case, case.formats[0])
    assert calls == [case.formats[0]]


def test_an_executor_steps_the_loaded_case_itself(tmp_path):
    """The Inspector's hook: the tool runs the loaded program, the runner does the rest."""
    manifest = write_scale(tmp_path, with_reference=False)
    case = next(c for c in expand(manifest) if c.name == "one_row")
    records, seen = [], []

    def step_all(context):
        seen.append(context)
        while len(records) < context.cycle_limit:
            records.append(context.model.step())
            if records[-1].outcome in ("halt", "fault"):
                return

    result = run_kernel(manifest, case, case.formats[0], RunOptions(executor=step_all))
    assert result.passed, result.describe()
    assert result.cycles == len(records) and records[-1].outcome == "halt"
    [context] = seen
    assert context.format is case.formats[0] and context.program.words
    assert "y" in context.placement.regions
    assert context.trace_header["format"] == context.format.value
    assert context.cycle_limit > 0


def test_an_executor_that_faults_reports_the_models_count_and_fault(tmp_path):
    """The runner takes the status and cycles from the model: the faulting word
    retires nothing, so it is not counted."""
    manifest = write_scale(tmp_path, with_reference=False)
    case = next(c for c in expand(manifest) if c.name == "default")
    result = run_kernel(
        manifest,
        case,
        ElementFormat.INT8,
        RunOptions(executor=lambda context: context.model.run(3)),  # F-9 at the 4th word
    )
    assert result.cycles == 3 and not result.passed
    assert result.fault is not None and result.fault.fault is FaultId.F9 and result.fault.pc == 3


@pytest.mark.parametrize("option", ["trace", "debug"])
def test_an_executor_runs_without_debug_or_trace(tmp_path, option):
    other = {"trace": tmp_path / "t.jsonl", "debug": DebugLaunch()}[option]
    with pytest.raises(ConfigurationError, match="executor"):
        RunOptions(executor=lambda context: None, **{option: other})
