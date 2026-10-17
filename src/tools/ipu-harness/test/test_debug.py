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
from ipu_emu.run_control import RunControl
from ipu_harness import cli, debug
from ipu_harness.cases import expand
from ipu_harness.debug import DebugHook, DebugLaunch
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
    before = set(sys.modules)
    assert run_kernel(scale, _default(scale), F.INT8).passed
    assert not {"curses", "ipu_debug"} & (set(sys.modules) - before)


def test_only_the_cli_reads_the_debug_environment():
    """In ipu_harness and ipu_kernels; the legacy ipu_apps readers go in E07-25."""
    scanned = {root: sorted(root.rglob("*.py")) for root in (HARNESS, KERNELS)}
    assert all(scanned.values()), "no sources found to scan"
    readers = sorted(
        str(path.relative_to(root.parent))
        for root, paths in scanned.items()
        for path in paths
        if "IPU_DEBUG_TUI" in path.read_text(encoding="utf-8")
    )
    assert readers == ["src/ipu_harness/cli.py"]


def _fake_debug_run(calls, *, quit_=False):
    """``ReferenceModel.debug_run`` as API 1.2.0 will have it: records its call,
    then runs (or reports a quit)."""

    def debug_run(model, callback, *, control=None, break_on_entry=False, max_cycles=None):
        calls.append((callback, control, break_on_entry, max_cycles))
        if quit_ == "interrupt":
            raise KeyboardInterrupt
        result = model.run(max_cycles)
        if quit_:
            return type("Quit", (), {"quit": True, "status": result.status})()
        return result

    return debug_run


def test_the_runner_passes_the_hook_and_stops_at_entry(scale, monkeypatch):
    calls = []
    monkeypatch.setattr(ReferenceModel, "debug_run", _fake_debug_run(calls), raising=False)
    control = RunControl()  # the debugger's own: its breakpoints live here
    launch = DebugLaunch(factory=lambda kernel: DebugHook(("callback for", kernel), control))
    options = RunOptions(debug=launch, max_cycles=500)
    result = run_kernel(scale, _default(scale), F.E4, options)
    assert result.passed, result.describe()
    assert calls == [(("callback for", "scale"), control, True, 500)]
    assert calls[0][1] is control


@pytest.mark.parametrize("how", [True, "interrupt"], ids=["quit", "ctrl-c"])
def test_quitting_the_debugger_ends_with_no_result(scale, monkeypatch, tmp_path, capsys, how):
    monkeypatch.setattr(ReferenceModel, "debug_run", _fake_debug_run([], quit_=how), raising=False)
    launch = DebugLaunch(factory=lambda kernel: DebugHook(None))
    result = run_kernel(scale, _default(scale), F.INT8, RunOptions(debug=launch))
    assert result.quit and not result.passed and result.comparisons == () and not result.outputs
    assert result.describe() == "scale/default/int8: QUIT  (no result)"
    # The CLI stops there, with exit code 0, before the next format.
    monkeypatch.setattr(cli, "DebugLaunch", lambda: launch)
    out = tmp_path / "out"
    assert cli.main([str(scale.directory), "--debug", "--format", "all", "--output", str(out)]) == 0
    assert capsys.readouterr().out.splitlines() == ["scale/default/fp32: QUIT  (no result)"]
    assert not out.exists()


def test_debug_and_trace_do_not_combine(tmp_path):
    with pytest.raises(ConfigurationError, match="--trace and --debug"):
        RunOptions(debug=DebugLaunch(), trace=tmp_path / "t.jsonl")


def test_a_model_without_the_calls_says_which_api_it_needs():
    with pytest.raises(ConfigurationError, match=r"ReferenceModel.debug_run \(.*1\.2\.0\)"):
        debug.debug_run(object(), DebugHook(None), break_on_entry=True, max_cycles=None)
    with pytest.raises(ConfigurationError, match=r"attach_alias_profile \(.*1\.2\.0\)"):
        debug.attach_alias_profile(object(), None)


def test_the_profile_attaches_to_the_one_run(scale, monkeypatch, tmp_path, capsys):
    attached = []
    monkeypatch.setattr(
        ReferenceModel, "attach_alias_profile", lambda m, p: attached.append(p), raising=False
    )
    result = run_kernel(scale, _default(scale), F.INT8, RunOptions(alias_profile=True))
    assert attached == [result.alias_profile]
    report = tmp_path / "aliases.json"
    # --alias-report alone profiles, and writes the run's profile.
    assert cli.main([str(scale.directory), "--alias-report", str(report)]) == 0
    assert len(attached) == 2
    metadata = json.loads(report.read_text())["metadata"]
    assert metadata == {"kernel": "scale", "case": "default", "format": "fp32"}


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
