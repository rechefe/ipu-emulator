"""Factory boundaries and the shared case lifecycle."""

import subprocess
import sys
from unittest.mock import Mock

import pytest

from ipu_apps.kernel_registry import create_harness, kernel_spec, kernels, resolve
from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernel_registry.cases import KernelCase, PreparedCase, load_cases, run_case
from ipu_apps.kernel_registry.runner import main
from ipu_emu.ipu_state import IpuState


def test_exact_selection_and_bindings(tmp_path):
    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(3 * 128 * 4))
    app = create_harness(
        "identity",
        params={"shape": (3, 128)},
        bindings={"inst_path": tmp_path / "inst.bin", "input_path": inp},
    )
    assert type(app) is kernel_spec("identity").app_class
    assert app.layout.input_rows == 3
    with pytest.raises(ValueError, match="unknown kernel"):
        create_harness("missing", params={}, bindings={})
    with pytest.raises(ValueError):
        create_harness("identity", params={"shape": (3, 127)}, bindings={})
    with pytest.raises(ValueError, match="invalid bindings"):
        create_harness(
            "identity", params={"shape": (3, 128)}, bindings={"shape": (1, 128), "inst_path": "x"}
        )
    with pytest.raises(ValueError, match="inst_path"):
        create_harness("identity", params={"shape": (3, 128)}, bindings={})


def test_resolved_kernel_uses_same_factory(tmp_path):
    verdict = resolve("softmax", shape=(2, 32), dim=1)
    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(2 * 32 * 4))
    app = create_harness(
        verdict.app_name,
        params={"shape": (2, 32), "dim": 1},
        bindings={"inst_path": "unused.bin", "input_path": inp},
    )
    assert type(app) is verdict.kernel.app_class
    assert app.n == 32
    with pytest.raises(ValueError):
        create_harness("softmax_rows", params={"shape": (2, 32), "dim": 1}, bindings={})


def test_state_factory_only_when_needed(monkeypatch):
    import ipu_apps.kernel_registry.base as base

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
    state, cycles = run_case(
        "identity", load_cases("identity")["default"], options={"rows": 1}, output_path=out
    )
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
assert not any(n.endswith(('.test', '.cases')) or '.test_' in n for n in sys.modules if n.startswith('ipu_apps.'))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frontend(capsys):
    assert main(["--kernel", "identity", "--list-cases"]) == 0
    assert "single_row" in capsys.readouterr().out
    assert main(["--kernel", "identity", "--case", "single_row", "--rows", "2"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity", "--case", "missing"])
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "fmt,quantize,row_bytes",
    [
        ("int8", False, 128),
        ("e4", False, 128),
        ("fp32", False, 512),
        ("int32", False, 512),
        ("fp32", True, 512),
    ],
)
def test_execution_profiles_and_fresh_state(monkeypatch, fmt, quantize, row_bytes):
    import ipu_apps.kernel_registry.registry as registry
    from ipu_apps.kernel_registry import ExecutionConfig, KernelSpec
    from ipu_emu.ipu import xmem_row_size_bytes
    from ipu_isa.formats import ElementFormat

    config = ExecutionConfig(fmt=ElementFormat(fmt), quantize_output=quantize)
    spec = KernelSpec(
        "test",
        "test",
        IpuApp,
        lambda **_: True,
        lambda **_: {},
        lambda **_: "test",
        execution=config,
    )
    monkeypatch.setattr(registry, "kernel_spec", lambda *_, **__: spec)
    app = create_harness("test", params={}, bindings={"inst_path": "unused"})
    first = app.make_state()
    assert first.fmt is ElementFormat(fmt)
    assert xmem_row_size_bytes(first) == row_bytes
    assert first.aaq.quantize_output is quantize
    first.regfile.set_lr(0, 42)
    first.xmem.write_address(0, b"changed")
    second = app.make_state()
    assert second is not first
    assert second.regfile.get_lr(0) == 0
    assert bytes(second.xmem.read_address(0, 7)) == bytes(7)


#: Kernels whose default case does not run in the wide-vector FP32 format.
_DEFAULT_FORMATS = {"fully_connected": "int32"}


@pytest.mark.parametrize("name", [spec.name for spec in kernels()])
def test_factory_and_direct_constructor_execution_agree(name, tmp_path):
    case = load_cases(name)["default"]
    prepared = case.prepare(tmp_path, **case.defaults)
    bindings = dict(prepared.bindings, inst_path="unused")
    spec = kernel_spec(name)
    via_factory = create_harness(name, params=prepared.params, bindings=bindings)
    direct = spec.app_class(**spec.build(**prepared.params), **bindings)
    for app in (via_factory, direct):
        state = app.make_state()
        assert not state.aaq.quantize_output
        assert state.fmt.value == _DEFAULT_FORMATS.get(name, "fp32")


def test_explicit_state_bypasses_execution_selector(monkeypatch):
    from dataclasses import replace

    import ipu_apps.kernel_registry.base as base
    import ipu_apps.kernel_registry.registry as registry

    def forbidden(app):
        pytest.fail("an explicit state must bypass execution configuration")

    spec = replace(kernel_spec("fully_connected"), execution=forbidden)
    monkeypatch.setattr(registry, "kernel_spec", lambda *_, **__: spec)
    app = create_harness(
        "fully_connected",
        params={"dtype": "INT8", "wide_mode": True},
        bindings={"inst_path": "unused", "inputs_path": "unused", "weights_path": "unused"},
    )
    supplied = IpuState()
    supplied.regfile.set_lr(0, 91)
    monkeypatch.setattr(base, "run_test", lambda **kw: kw["state"])
    assert app.run(state=supplied) is supplied
    assert supplied.fmt.value == "int8"
    assert supplied.regfile.get_lr(0) == 91


def test_unregistered_harness_defaults_and_direct_subclass(monkeypatch, tmp_path):
    import ipu_apps.kernel_registry.registry as registry
    from ipu_apps.kernels.reshape.identity.app import IdentityApp

    def forbidden(*args, **kwargs):
        pytest.fail("direct construction must not scan the registry")

    monkeypatch.setattr(registry, "load", forbidden)
    plain = IpuApp(inst_path="unused").make_state()
    assert plain.fmt.value == "int8"

    class DerivedIdentity(IdentityApp):
        pass

    inp = tmp_path / "input.bin"
    inp.write_bytes(bytes(512))
    app = DerivedIdentity(inst_path="unused", input_path=inp, params={"shape": (1, 128)})
    assert app.make_state().fmt.value == "fp32"


def test_execution_config_is_immutable_and_rejects_bad_formats():
    from dataclasses import FrozenInstanceError

    from ipu_apps.kernel_registry import ExecutionConfig
    from ipu_emu.errors import ConfigurationError
    from ipu_isa.formats import ElementFormat

    config = ExecutionConfig()
    with pytest.raises(FrozenInstanceError):
        config.fmt = ElementFormat.FP32
    with pytest.raises(ConfigurationError, match="execution format"):
        ExecutionConfig(fmt="typo")


def test_frontend_help_and_case_listing(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--output" in capsys.readouterr().out
    assert main(["--kernel", "identity", "--list-cases", "--case", "missing"]) == 0
    assert "single_row" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity", "--help"])
    assert exc.value.code == 0
    assert "--rows" in capsys.readouterr().out


@pytest.mark.parametrize(
    "defaults",
    [
        {"output": "x.bin"},
        {"max_cycles": 4},
        {"bad-name": 1},
        {"value": None},
        {"shape": (3, 128)},
        {1: "bad"},
    ],
)
def test_invalid_case_option_declarations(defaults):
    with pytest.raises(ValueError, match="case option"):
        KernelCase(lambda _: None, defaults)


def test_frontend_boolean_options_and_collisions(monkeypatch, capsys):
    from types import SimpleNamespace

    import ipu_apps.kernel_registry.runner as runner

    options = []
    state = SimpleNamespace(quit_requested=False, stats=SimpleNamespace(format_summary=lambda: ""))
    monkeypatch.setattr(
        runner, "run_case", lambda *_, **kw: (options.append(kw["options"]) or state, 1)
    )
    monkeypatch.setattr(
        runner, "load_cases", lambda _: {"default": KernelCase(lambda _: None, {"enabled": True})}
    )
    assert main(["--kernel", "identity", "--no-enabled"]) == 0
    assert options == [{"enabled": False}]
    monkeypatch.setattr(
        runner,
        "load_cases",
        lambda _: {"default": KernelCase(lambda _: None, {"enabled": True, "no_enabled": False})},
    )
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity"])
    assert exc.value.code == 1
    assert "conflicting option" in capsys.readouterr().err


def test_missing_case_declaration_is_actionable(monkeypatch, capsys):
    from types import SimpleNamespace

    import ipu_apps.kernel_registry.cases as cases

    monkeypatch.setattr(cases, "import_module", lambda _: SimpleNamespace())
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity"])
    assert exc.value.code == 1
    assert (
        "cannot load CASES from ipu_apps.kernels.reshape.identity.cases" in capsys.readouterr().err
    )


def test_failed_output_is_exported_with_diagnostics(tmp_path, monkeypatch, capsys):
    from ipu_apps.kernels.reshape.identity.app import IdentityApp

    monkeypatch.setattr(
        IdentityApp, "teardown", lambda app, state: app.output_path.write_bytes(b"bad")
    )
    out = tmp_path / "failed.bin"
    with pytest.raises(SystemExit) as exc:
        main(["--kernel", "identity", "--output", str(out)])
    assert exc.value.code == 1
    assert out.read_bytes() == b"bad"
    assert "output size mismatch" in capsys.readouterr().err


def test_relative_output_paths_follow_the_bazel_run_directory(tmp_path, monkeypatch, capsys):
    """`bazel run` executes inside runfiles; relative --output/--alias-report
    must still land where the user ran the command."""
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))
    assert (
        main(["--kernel", "identity", "--output", "out.bin", "--alias-report", "aliases.json"]) == 0
    )
    assert (tmp_path / "out.bin").stat().st_size > 0
    assert (tmp_path / "aliases.json").read_text().startswith("{")
    absolute = tmp_path / "elsewhere" / "abs.bin"
    absolute.parent.mkdir()
    assert main(["--kernel", "identity", "--output", str(absolute)]) == 0
    assert absolute.exists()


def test_runtime_checks_survive_optimization():
    code = """
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_emu.errors import DebugOnlyInstructionWarning
for name in ('identity', 'softmax_rows', 'fully_connected'):
    with TemporaryDirectory() as tmp:
        case = load_cases(name)['default']
        prepared = case.prepare(Path(tmp), **case.defaults)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', DebugOnlyInstructionWarning)
            run_case(name, case, workspace=Path(tmp))
        # Only fully_connected uses the emulator-only ST.ACC.
        debug_only = any(issubclass(w.category, DebugOnlyInstructionWarning) for w in caught)
        if debug_only != (name == 'fully_connected'):
            raise SystemExit(name + ': unexpected DebugOnlyInstructionWarning state')
        out = Path(prepared.bindings['output_path'])
        raw = bytearray(out.read_bytes())
        if name == 'softmax_rows':
            raw[:] = bytes(len(raw))
        else:
            raw[0] ^= 255
        out.write_bytes(raw)
        try:
            prepared.check()
        except ValueError as exc:
            if not str(exc):
                raise SystemExit('missing diagnostic')
        else:
            raise SystemExit(name + ' accepted corrupted output')
"""
    subprocess.run([sys.executable, "-O", "-c", code], check=True)


def test_cases_do_not_import_pytest():
    code = """
import sys
from ipu_apps.kernel_registry import kernels
from ipu_apps.kernel_registry.cases import load_cases
for spec in kernels():
    if spec.name in ('identity', 'fully_connected') or spec.op == 'softmax':
        load_cases(spec.name)
if 'pytest' in sys.modules:
    raise SystemExit('runtime cases imported pytest')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cases_and_assembly_follow_package_not_class_module(tmp_path, monkeypatch):
    from importlib import import_module
    from importlib.resources import files

    import ipu_apps.kernel_registry.cases as cases
    import ipu_apps.kernel_registry.registry as registry

    package = tmp_path / "split_kernel"
    package.mkdir()
    (package / "app.py").write_text(
        "from ipu_apps.kernels.reshape.identity.app import IdentityApp\n"
        "class SplitApp(IdentityApp): pass\n"
    )
    (package / "__init__.py").write_text(
        "from dataclasses import replace\n"
        "from ipu_apps.kernels.reshape.identity.app import SPEC as BASE\n"
        "from .app import SplitApp\n"
        "SPEC = replace(BASE, name='split_identity', app_class=SplitApp, asm='copy.asm')\n"
    )
    (package / "cases.py").write_text("from ipu_apps.kernels.reshape.identity.cases import CASES\n")
    (package / "copy.asm").write_text(
        files("ipu_apps.kernels.reshape.identity").joinpath("identity.asm").read_text()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = import_module("split_kernel").SPEC
    monkeypatch.setattr(registry, "kernel_spec", lambda *_, **__: spec)
    monkeypatch.setattr(cases, "kernel_spec", lambda *_, **__: spec)
    assert spec.resource_package == "split_kernel"
    state, cycles = run_case("split_identity", load_cases("split_identity")["default"])
    assert state.is_halted and cycles > 0


def test_pytest_config_collects_adjacent_suites(tmp_path):
    import shutil
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "pyproject.toml"
    shutil.copyfile(config, tmp_path / "pyproject.toml")
    for package in ("test", "src/first", "src/second"):
        directory = tmp_path / package
        directory.mkdir(parents=True)
        name = "test_old.py" if package == "test" else "test.py"
        (directory / name).write_text("def test_example(): pass\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 tests collected" in result.stdout


def test_frontend_alias_report(tmp_path):
    import json

    from ipu_apps.kernel_registry.runner import main

    report = tmp_path / "aliases.json"
    assert main(["--kernel", "identity", "--alias-report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["schema_version"] == 1
    assert data["metadata"]["kernel"] == "identity"
    assert data["aliases"]["A1_MOV_RC"]["measurements"]


def test_every_assembly_has_a_registered_case():
    from pathlib import Path

    from ipu_apps.kernel_registry import kernels

    root = Path(__file__).resolve().parents[1] / "src/ipu_apps"
    assemblies = {p.stem for p in root.rglob("*.asm")}
    registered = {spec.name for spec in kernels()}
    assert assemblies == registered
    for name in sorted(registered):
        assert "default" in load_cases(name)


def test_memory_harness_rejects_truncated_image(tmp_path):
    inp = tmp_path / "input.bin"
    inp.write_bytes(b"short")
    with pytest.raises(ValueError, match="preformatted input must contain"):
        create_harness(
            "channel_peak",
            params=dict(shape=(2, 3)),
            bindings=dict(inst_path="missing", input_path=inp),
        )
