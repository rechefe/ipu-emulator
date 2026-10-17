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
from ipu_emu.reference import RunResult as ModelRun
from ipu_harness import params
from ipu_harness.cases import expand
from ipu_harness.runner import RunOptions, run_kernel


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
        seen.append((context.plan.format, context.program.words, context.placement))
        while True:
            record = context.model.step()
            records.append(record)
            if record.outcome in ("halt", "fault"):
                return ModelRun(context.model.status, len(records))

    result = run_kernel(manifest, case, case.formats[0], RunOptions(executor=step_all))
    assert result.passed, result.describe()
    assert result.cycles == len(records) and records[-1].outcome == "halt"
    [(fmt, words, placement)] = seen
    assert fmt is case.formats[0] and words and "y" in placement.regions


def test_an_executor_runs_without_debug_or_trace(tmp_path):
    with pytest.raises(ConfigurationError, match="executor"):
        RunOptions(executor=lambda context: None, trace=tmp_path / "t.jsonl")
