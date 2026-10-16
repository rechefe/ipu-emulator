"""The per-case kernel baseline (E02-07): what a record captures and how a
change shows up, on the identity kernel and its recorded baseline."""

from __future__ import annotations

import json

import pytest

from ipu_apps.kernel_registry import baseline
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry.registry import kernel_spec
from ipu_apps.kernel_registry.testing import check_baseline

_KERNEL = "identity"


def _record(**options) -> baseline.CaseRecord:
    recorder = baseline.Recorder()
    run_case(_KERNEL, load_cases(_KERNEL)["default"], options=options, recorder=recorder)
    return recorder.record


def test_the_recorded_run_matches():
    record = _record()
    assert len(record.stages) == 1
    baseline.check(_KERNEL, "default", record)


def _wrap_setup(monkeypatch, after):
    """Run ``after(state)`` right after the identity harness's own setup."""
    from ipu_apps.kernel_registry.base import IpuApp

    original = IpuApp.run

    def run(self, **kwargs):
        setup = self.setup

        def wrapped(state):
            setup(state)
            after(state)

        self.setup = wrapped
        return original(self, **kwargs)

    monkeypatch.setattr(IpuApp, "run", run)


def test_a_changed_cr_fails_naming_the_register(monkeypatch):
    _wrap_setup(monkeypatch, lambda state: state.regfile.set_cr(9, 7))
    with pytest.raises(AssertionError, match=r"CR9: 0 -> 7"):
        baseline.check(_KERNEL, "default", _record())


def test_a_changed_input_fails_the_setup_digest(monkeypatch):
    # A byte the kernel never reads, so its own check still passes.
    def poke(state):
        state.xmem.write_address(XMEM_SIZE_BYTES - 1, b"\x01")

    _wrap_setup(monkeypatch, poke)
    with pytest.raises(AssertionError, match="xmem_setup_sha256"):
        baseline.check(_KERNEL, "default", _record())


def test_a_case_without_a_record_names_the_update_command():
    with pytest.raises(AssertionError, match="update_kernel_baselines"):
        baseline.check(_KERNEL, "no_such_case", _record())


def test_differences_list_each_field():
    stage = baseline.StageRecord("p", [0] * 16, [0] * 16, "x")
    old = baseline.CaseRecord([stage], "o", "s", 10)
    new = baseline.CaseRecord(
        [baseline.StageRecord("p", [0] * 15 + [1], [2] + [0] * 15, "y")], None, "s", 11
    )
    assert baseline.differences(old, new) == [
        "xmem_setup_sha256: x -> y",
        "CR15: 0 -> 1",
        "LR0: 0 -> 2",
        "output_sha256: o -> None",
        "cycles: 10 -> 11",
    ]


def test_the_update_command_writes_the_kernel_folder(monkeypatch, tmp_path):
    monkeypatch.setenv(baseline.UPDATE_ENV, str(tmp_path))
    path = baseline.source_path(tmp_path, _KERNEL)
    path.parent.mkdir(parents=True)
    record = _record()
    updated: dict = {}
    check_baseline(_KERNEL, "default", record, updated)
    assert path.as_posix().endswith("kernels/reshape/identity/baseline.json")
    written = json.loads(path.read_text())
    assert baseline.CaseRecord.from_json(written["default"]) == record
