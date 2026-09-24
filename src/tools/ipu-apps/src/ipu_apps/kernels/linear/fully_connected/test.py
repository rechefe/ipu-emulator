"""Fully connected regression tests using the runtime cases."""
import pytest

from ipu_apps.kernels.linear.fully_connected.cases import CASES, DATA, MissingInputFixture, prepare
from ipu_apps.kernel_registry import create_harness, resolve
from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__)


def _run(name):
    try:
        state, cycles = run_case("fully_connected", CASES[name])
    except MissingInputFixture as exc:
        pytest.skip(str(exc))
    assert cycles > 0
    return state


@pytest.mark.parametrize("name", CASES)
def test_mac_accounting_and_alias_hits(name):
    state = _run(name)
    # The kernel pads 64 outputs to 128 lanes and declares no narrower width,
    # so every multiply bills all 128 and the summary flags the overstatement.
    assert state.stats.mult_lane_ops == state.stats.mult_active_cycles * 128
    assert state.stats.alias_hits == {"A16_STR_ACC_REG": 10}


def test_missing_input_fixture_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setitem(prepare.__globals__, "DATA", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="missing fully connected fixture"):
        _run("int8")


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
    from ipu_emu.ipu_math import DType

    verdict = resolve("fully_connected", shape=(10, 128))
    assert verdict.supported
    bindings = {"inst_path": "unused", "inputs_path": "unused", "weights_path": "unused"}
    assert create_harness("fully_connected", params={}, bindings=bindings).dtype is DType.INT8
    app = FullyConnectedApp(dtype=4, **bindings)
    assert app.dtype is DType.E4
    assert app.make_state().dtype is DType.E4
