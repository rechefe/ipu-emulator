"""Fully connected regression tests using the runtime cases.

The kernel moved to //kernels/linear/fully_connected (E07-21), whose
test_accounting.py holds the MAC and alias accounting and the missing-file
test; this frozen copy stays until E07-25/E07-26.
"""

import pytest

from ipu_apps.kernel_registry import create_harness, resolve
from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.linear.fully_connected.cases import CASES, DATA, prepare
from ipu_emu.errors import DebugOnlyInstructionWarning

test_case = case_tests(__package__)


def _run(name):
    # The kernel stores each sample with the emulator-only ST.ACC.
    with pytest.warns(DebugOnlyInstructionWarning):
        state, cycles = run_case("fully_connected", CASES[name])
    assert cycles > 0
    return state


def test_missing_output_is_not_skipped(monkeypatch):
    from ipu_apps.kernels.linear.fully_connected.app import FullyConnectedApp

    monkeypatch.setattr(FullyConnectedApp, "teardown", lambda self, state: None)
    with pytest.raises(FileNotFoundError, match="output.bin"):
        _run("int8")


def test_missing_golden_is_not_skipped(monkeypatch, tmp_path):
    directory = tmp_path / "int8"
    directory.mkdir()
    for name in ("inputs_int8.bin", "weights_int8.bin"):
        (directory / name).write_bytes((DATA / "int8" / name).read_bytes())
    monkeypatch.setitem(prepare.__globals__, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="out_int8_acc_int32.bin"):
        _run("int8")


def test_dtype_default_and_normalization():
    from ipu_apps.kernels.linear.fully_connected.app import FullyConnectedApp
    from ipu_isa.formats import ElementFormat

    verdict = resolve("fully_connected", shape=(10, 128))
    assert verdict.supported
    bindings = {"inst_path": "unused", "inputs_path": "unused", "weights_path": "unused"}
    assert (
        create_harness("fully_connected", params={}, bindings=bindings).dtype is ElementFormat.INT8
    )
    app = FullyConnectedApp(dtype=4, **bindings)
    assert app.dtype is ElementFormat.E4
    assert app.make_state().fmt is ElementFormat.E4
    wide = FullyConnectedApp(dtype="int8", wide_mode=True, **bindings)
    assert wide.make_state().fmt is ElementFormat.INT32
