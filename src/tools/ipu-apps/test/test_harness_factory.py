"""Factory boundaries and the shared case lifecycle."""
import subprocess
import sys
from unittest.mock import Mock

import pytest

from ipu_apps.base import IpuApp
from ipu_apps.kernel_registry import create_harness, kernel_spec, resolve
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, load_cases, run_case
from ipu_apps.kernel_registry.runner import main
from ipu_emu.ipu_state import IpuState


def test_exact_selection_and_bindings(tmp_path):
    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(3 * 128 * 4))
    app = create_harness("identity", params={"shape": (3, 128)},
                         bindings={"inst_path": tmp_path / "inst.bin", "input_path": inp})
    assert type(app) is kernel_spec("identity").app_class
    assert app.rows == 3
    with pytest.raises(ValueError, match="unknown kernel"):
        create_harness("missing", params={}, bindings={})
    with pytest.raises(ValueError):
        create_harness("identity", params={"shape": (3, 127)}, bindings={})
    with pytest.raises(ValueError, match="invalid bindings"):
        create_harness("identity", params={"shape": (3, 128)},
                       bindings={"shape": (1, 128), "inst_path": "x"})
    with pytest.raises(ValueError, match="inst_path"):
        create_harness("identity", params={"shape": (3, 128)}, bindings={})


def test_resolved_kernel_uses_same_factory(tmp_path):
    verdict = resolve("softmax", shape=(2, 32), dim=1)
    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(2 * 32 * 4))
    app = create_harness(verdict.app_name, params={"shape": (2, 32), "dim": 1},
                         bindings={"inst_path": "unused.bin", "input_path": inp})
    assert type(app) is verdict.kernel.app_class
    assert app.n == 32
    with pytest.raises(ValueError):
        create_harness("softmax_rows", params={"shape": (2, 32), "dim": 1}, bindings={})


def test_state_factory_only_when_needed(monkeypatch):
    import ipu_apps.base as base
    supplied = IpuState()
    app = IpuApp(inst_path="unused")
    app.make_state = Mock(return_value=IpuState())
    monkeypatch.setattr(base, "run_test", lambda **kw: kw["state"])
    assert app.run(state=supplied) is supplied
    app.make_state.assert_not_called()
    assert app.run() is app.make_state.return_value
    app.make_state.assert_called_once()


def test_case_execution_and_export(tmp_path):
    out = tmp_path / "copy.bin"
    state, cycles = run_case("identity", load_cases("identity")["default"],
                             options={"rows": 1}, output_path=out)
    assert state.is_halted and cycles > 0
    assert len(out.read_bytes()) == 512
    with pytest.raises(ValueError, match="unknown case options"):
        run_case("identity", load_cases("identity")["default"], options={"typo": 1})
    with pytest.raises(ValueError, match="max_cycles"):
        run_case("identity", load_cases("identity")["default"], max_cycles=0)


def test_failure_cleanup(monkeypatch):
    paths = []
    original = load_cases("identity")["default"]
    def prepare(workspace):
        paths.append(workspace)
        prepared = original.prepare(workspace, rows=1)
        def fail():
            raise AssertionError("bad output")
        return PreparedCase(prepared.params, prepared.bindings, fail)
    with pytest.raises(AssertionError, match="bad output"):
        run_case("identity", KernelCase(prepare))
    assert not paths[0].exists()


def test_discovery_does_not_import_cases():
    code = """
import sys
from ipu_apps.kernel_registry import discover
found = discover()
assert any(s.name == 'identity' for s in found.specs)
assert not any(n.endswith('.test') or '.test_' in n for n in sys.modules if n.startswith('ipu_apps.'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frontend(capsys):
    assert main(["--kernel", "identity", "--list-cases"]) == 0
    assert "single_row" in capsys.readouterr().out
    assert main(["--kernel", "identity", "--case", "single_row", "--rows", "2"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity", "--case", "missing"])
    assert exc.value.code == 1


@pytest.mark.parametrize("mode,dtype,quantize,row_bytes", [
    ("native", "INT8", False, 128),
    ("native", "E4", False, 128),
    ("fp32", "INT8", False, 512),
    ("int32", "INT8", False, 512),
    ("fp32", "INT8", True, 512),
])
def test_execution_profiles_and_fresh_state(monkeypatch, mode, dtype, quantize, row_bytes):
    from ipu_apps.kernel_registry import ExecutionConfig, KernelSpec
    import ipu_apps.kernel_registry.registry as registry
    from ipu_emu.ipu import xmem_row_size_bytes
    from ipu_emu.ipu_math import DType

    config = ExecutionConfig(mode=mode, dtype=DType[dtype], quantize_output=quantize)
    spec = KernelSpec("test", "test", IpuApp, lambda **_: True, lambda **_: {},
                      lambda **_: "test", execution=config)
    monkeypatch.setattr(registry, "kernel_spec", lambda *_, **__: spec)
    app = create_harness("test", params={}, bindings={"inst_path": "unused"})
    first = app.make_state()
    assert first.dtype == DType[dtype]
    assert xmem_row_size_bytes(first) == row_bytes
    assert first.wide_vector_debug == (mode != "native")
    assert first.wide_vector_arithmetic.value == ("int32" if mode == "int32" else "fp32")
    assert first.wide_vector_quantize_output is quantize
    first.regfile.set_lr(0, 42)
    first.xmem.write_address(0, b"changed")
    second = app.make_state()
    assert second is not first
    assert second.regfile.get_lr(0) == 0
    assert bytes(second.xmem.read_address(0, 7)) == bytes(7)


@pytest.mark.parametrize("name", [
    "identity", "softmax_rows", "softmax_rows_partial", "softmax_rows_long",
    "softmax_columns", "softmax_columns_packed", "fully_connected",
])
def test_factory_and_direct_constructor_execution_agree(name, tmp_path):
    case = load_cases(name)["default"]
    prepared = case.prepare(tmp_path, **case.defaults)
    bindings = dict(prepared.bindings, inst_path="unused")
    spec = kernel_spec(name)
    via_factory = create_harness(name, params=prepared.params, bindings=bindings)
    direct = spec.app_class(**spec.build(**prepared.params), **bindings)
    for app in (via_factory, direct):
        state = app.make_state()
        assert state.wide_vector_debug
        assert not state.wide_vector_quantize_output
        assert state.wide_vector_arithmetic.value == ("int32" if name == "fully_connected" else "fp32")


def test_explicit_state_bypasses_execution_selector(monkeypatch):
    from dataclasses import replace
    import ipu_apps.base as base
    import ipu_apps.kernel_registry.registry as registry

    def forbidden(app):
        pytest.fail("an explicit state must bypass execution configuration")

    spec = replace(kernel_spec("fully_connected"), execution=forbidden)
    monkeypatch.setattr(registry, "kernel_spec", lambda *_, **__: spec)
    app = create_harness("fully_connected", params={"dtype": "INT8", "wide_mode": True},
                         bindings={"inst_path": "unused", "inputs_path": "unused", "weights_path": "unused"})
    supplied = IpuState()
    supplied.regfile.set_lr(0, 91)
    monkeypatch.setattr(base, "run_test", lambda **kw: kw["state"])
    assert app.run(state=supplied) is supplied
    assert not supplied.wide_vector_debug
    assert supplied.regfile.get_lr(0) == 91


def test_unregistered_harness_defaults_and_direct_subclass(monkeypatch, tmp_path):
    import ipu_apps.kernel_registry.registry as registry
    from ipu_apps.kernel_registry.identity import IdentityApp

    def forbidden(*args, **kwargs):
        pytest.fail("direct construction must not scan the registry")
    monkeypatch.setattr(registry, "load", forbidden)
    plain = IpuApp(inst_path="unused").make_state()
    assert not plain.wide_vector_debug
    class DerivedIdentity(IdentityApp):
        pass
    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(512))
    app = DerivedIdentity(inst_path="unused", input_path=inp)
    assert app.make_state().wide_vector_debug


def test_execution_config_is_immutable_and_rejects_bad_modes():
    from dataclasses import FrozenInstanceError
    from ipu_apps.kernel_registry import ExecutionConfig

    config = ExecutionConfig()
    with pytest.raises(FrozenInstanceError):
        config.mode = "fp32"
    with pytest.raises(ValueError, match="execution mode"):
        ExecutionConfig(mode="typo")
