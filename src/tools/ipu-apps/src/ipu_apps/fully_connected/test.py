"""Fully connected fixtures and checks, shared by run and test."""
from pathlib import Path

import pytest

from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, run_case

DATA = Path(__file__).with_name("test_data_format")


class MissingInputFixture(FileNotFoundError):
    """An optional input or weight fixture is unavailable."""


def prepare(workspace, *, dtype, wide_mode=False):
    name = dtype.lower()
    if name not in ("int8", "fp8_e4m3", "fp8_e5m2"):
        raise ValueError("dtype must be INT8, FP8_E4M3, or FP8_E5M2")
    directory = DATA / name
    inputs = directory / f"inputs_{name}.bin"
    weights = directory / f"weights_{name}.bin"
    for path in (inputs, weights):
        if not path.is_file():
            raise MissingInputFixture(f"missing fully connected fixture: {path}")
    out = workspace / "output.bin"
    suffix = "int32" if name == "int8" else "fp32"
    golden = directory / f"out_{name}_acc_{suffix}.bin"

    def check():
        assert out.read_bytes() == golden.read_bytes()

    return PreparedCase({"dtype": dtype, "wide_mode": wide_mode},
                        {"inputs_path": inputs, "weights_path": weights, "output_path": out}, check)


def prepare_default(workspace, *, dtype):
    # Retain the existing INT8 runner's wide arithmetic configuration.
    return prepare(workspace, dtype=dtype, wide_mode=dtype.upper() == "INT8")


CASES = {
    "default": KernelCase(prepare_default, {"dtype": "INT8"}, 2_000_000),
    "int8": KernelCase(prepare, {"dtype": "INT8"}, 2_000_000),
    "fp8_e4m3": KernelCase(prepare, {"dtype": "FP8_E4M3"}, 2_000_000),
    "fp8_e5m2": KernelCase(prepare, {"dtype": "FP8_E5M2"}, 2_000_000),
}


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
