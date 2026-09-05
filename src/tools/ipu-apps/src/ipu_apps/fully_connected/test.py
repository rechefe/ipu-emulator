"""Fully connected regression tests using the runtime cases."""
import pytest

from ipu_apps.fully_connected.cases import CASES, DATA, MissingInputFixture, prepare
from ipu_apps.kernel_registry.cases import run_case


@pytest.mark.parametrize("name", CASES)
def test_fc(name):
    try:
        _, cycles = run_case("fully_connected", CASES[name])
    except MissingInputFixture as exc:
        pytest.skip(str(exc))
    assert cycles > 0


def test_missing_input_fixture_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setitem(prepare.__globals__, "DATA", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="missing fully connected fixture"):
        test_fc("int8")


def test_missing_output_is_not_skipped(monkeypatch):
    from ipu_apps.fully_connected import FullyConnectedApp

    monkeypatch.setattr(FullyConnectedApp, "teardown", lambda self, state: None)
    with pytest.raises(FileNotFoundError, match="output.bin"):
        test_fc("int8")


def test_missing_golden_is_not_skipped(monkeypatch, tmp_path):
    directory = tmp_path / "int8"
    directory.mkdir()
    for name in ("inputs_int8.bin", "weights_int8.bin"):
        (directory / name).write_bytes((DATA / "int8" / name).read_bytes())
    monkeypatch.setitem(prepare.__globals__, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="out_int8_acc_int32.bin"):
        test_fc("int8")
