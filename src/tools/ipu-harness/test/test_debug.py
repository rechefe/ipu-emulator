"""Debugging and alias profiling are options of the one run a case makes (E06-12).

They run on the reference-model API 1.3.0 (``ReferenceModel.debug_run`` and
``attach_alias_profile``) with callbacks standing in for the TUI; a spy around
``debug_run`` shows what the runner passes it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scale_kernel import write_scale

from ipu_emu.emulator import DebugAction
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_emu.run_control import RunControl
from ipu_harness import cli
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


def _spy(monkeypatch):
    """Record each ``debug_run`` call's arguments, then run the real one."""
    calls = []
    real = ReferenceModel.debug_run

    def debug_run(model, callback, break_on_entry=False, *, max_cycles=None, control=None):
        calls.append((callback, control, break_on_entry, max_cycles))
        return real(model, callback, break_on_entry, max_cycles=max_cycles, control=control)

    monkeypatch.setattr(ReferenceModel, "debug_run", debug_run)
    return calls


def _callback(action):
    stops = []

    def callback(state, cycle):
        stops.append(cycle)
        return action

    callback.stops = stops
    return callback


def test_the_runner_passes_the_hook_and_stops_at_entry(scale, monkeypatch):
    calls = _spy(monkeypatch)
    callback = _callback(DebugAction.CONTINUE)
    control = RunControl()  # the debugger's own: its breakpoints live here
    launch = DebugLaunch(factory=lambda kernel: DebugHook(callback, control))
    result = run_kernel(scale, _default(scale), F.E4, RunOptions(debug=launch, max_cycles=500))
    assert result.passed, result.describe()
    assert calls == [(callback, control, True, 500)]
    assert callback.stops[0] == 0  # the entry stop, before the first word


def test_quitting_the_debugger_ends_with_no_result(scale, monkeypatch, tmp_path, capsys):
    launch = DebugLaunch(factory=lambda kernel: DebugHook(_callback(DebugAction.QUIT)))
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


def test_the_profile_attaches_to_the_one_run(scale, monkeypatch, tmp_path, capsys):
    attached = []
    real = ReferenceModel.attach_alias_profile

    def attach(model, profile):
        attached.append(profile)
        real(model, profile)

    monkeypatch.setattr(ReferenceModel, "attach_alias_profile", attach)
    result = run_kernel(scale, _default(scale), F.INT8, RunOptions(alias_profile=True))
    assert attached == [result.alias_profile]
    report = tmp_path / "aliases.json"
    # --alias-report alone profiles, and writes the run's profile.
    assert cli.main([str(scale.directory), "--alias-report", str(report)]) == 0
    assert len(attached) == 2
    metadata = json.loads(report.read_text())["metadata"]
    # The runner names the run; the model adds its own fields.
    assert metadata.items() >= {"kernel": "scale", "case": "default", "format": "fp32"}.items()


@pytest.mark.parametrize("fmt", [F.FP32, F.INT8, F.E4])
def test_a_profiled_run_equals_a_plain_one(scale, fmt):
    case = _default(scale)
    plain = run_kernel(scale, case, fmt)
    profiled = run_kernel(scale, case, fmt, RunOptions(alias_profile=True))
    assert profiled.cycles == plain.cycles == profiled.alias_profile.cycles
    assert profiled.raw_outputs == plain.raw_outputs
    np.testing.assert_array_equal(profiled.outputs["y"], plain.outputs["y"])
