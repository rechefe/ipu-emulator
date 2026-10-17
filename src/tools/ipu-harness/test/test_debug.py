"""Debugging and alias profiling are options of the one run a case makes (E06-12).

The reference-model calls behind them (``ReferenceModel.debug_run`` and
``attach_alias_profile``, API 1.2.0) are stood in for by fakes, so these tests
check the harness's wiring. The profiled-versus-plain equivalence runs on the
real API once the model has it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scale_kernel import write_scale

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness import cli, debug
from ipu_harness.cases import expand
from ipu_harness.debug import DebugLaunch
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat as F

HARNESS = Path(__file__).resolve().parents[1] / "src"
KERNELS = Path(__file__).resolve().parents[2] / "ipu-kernels" / "src"


@pytest.fixture
def scale(tmp_path):
    return write_scale(tmp_path, with_reference=False)


def _default(manifest):
    return next(case for case in expand(manifest) if case.name == "default")


def test_debug_and_the_environment_both_make_a_launch():
    assert cli.debug_launch(True, {}) == DebugLaunch()
    assert cli.debug_launch(False, {"IPU_DEBUG_TUI": "1"}) == DebugLaunch()
    assert cli.debug_launch(False, {"IPU_DEBUG_TUI": "0"}) is None
    assert cli.debug_launch(False, {}) is None


def test_a_plain_run_imports_no_debugger(scale):
    assert run_kernel(scale, _default(scale), F.INT8).passed
    assert "curses" not in sys.modules
    assert "ipu_debug" not in sys.modules


def test_only_the_cli_reads_the_debug_environment():
    """In ipu_harness and ipu_kernels; the legacy ipu_apps readers go in E07-25."""
    readers = sorted(
        str(path.relative_to(root.parent))
        for root in (HARNESS, KERNELS)
        for path in root.rglob("*.py")
        if "IPU_DEBUG_TUI" in path.read_text(encoding="utf-8")
    )
    assert readers == ["src/ipu_harness/cli.py"]


def _fake_debug_run(calls):
    """``ReferenceModel.debug_run`` as API 1.2.0 will have it: records its call, then runs."""

    def debug_run(model, callback, *, break_on_entry=False):
        calls.append((callback, break_on_entry))
        return model.run()

    return debug_run


def test_the_runner_passes_the_debug_callback_and_stops_at_entry(scale, monkeypatch):
    calls = []
    monkeypatch.setattr(ReferenceModel, "debug_run", _fake_debug_run(calls), raising=False)
    launch = DebugLaunch(factory=lambda kernel: ("callback for", kernel))
    result = run_kernel(scale, _default(scale), F.E4, RunOptions(debug=launch))
    assert result.passed, result.describe()
    assert calls == [(("callback for", "scale"), True)]


def test_debug_and_trace_do_not_combine(scale, monkeypatch, tmp_path):
    monkeypatch.setattr(ReferenceModel, "debug_run", _fake_debug_run([]), raising=False)
    options = RunOptions(debug=DebugLaunch(factory=lambda k: None), trace=tmp_path / "t.jsonl")
    with pytest.raises(ConfigurationError, match="--trace and --debug"):
        run_kernel(scale, _default(scale), F.INT8, options)


def test_a_model_without_the_calls_says_which_api_it_needs():
    with pytest.raises(ConfigurationError, match=r"ReferenceModel.debug_run \(.*1\.2\.0\)"):
        debug.debug_run(object(), None, break_on_entry=True)
    with pytest.raises(ConfigurationError, match=r"attach_alias_profile \(.*1\.2\.0\)"):
        debug.attach_alias_profile(object(), None)


def test_the_profile_attaches_to_the_one_run(scale, monkeypatch, tmp_path, capsys):
    attached = []
    monkeypatch.setattr(
        ReferenceModel, "attach_alias_profile", lambda m, p: attached.append(p), raising=False
    )
    result = run_kernel(scale, _default(scale), F.INT8, RunOptions(alias_profile=True))
    assert attached == [result.alias_profile] and result.alias_profile is not None
    report = tmp_path / "aliases.json"
    assert cli.main([str(scale.directory), "--alias-report", str(report)]) == 0
    assert "aliases" in json.loads(report.read_text())


@pytest.mark.skipif(
    not hasattr(ReferenceModel, "attach_alias_profile"),
    reason="needs ReferenceModel.attach_alias_profile (reference-model API 1.2.0, lane C)",
)
@pytest.mark.parametrize("fmt", [F.FP32, F.INT8, F.E4])
def test_a_profiled_run_equals_a_plain_one(scale, fmt):
    case = _default(scale)
    plain = run_kernel(scale, case, fmt)
    profiled = run_kernel(scale, case, fmt, RunOptions(alias_profile=True))
    assert profiled.cycles == plain.cycles == profiled.alias_profile.cycles
    assert profiled.raw_outputs == plain.raw_outputs
    np.testing.assert_array_equal(profiled.outputs["y"], plain.outputs["y"])
