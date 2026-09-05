"""Reusable cases and execution through the kernel registry's harness factory."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from importlib.resources import files
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

from ipu_as.lark_tree import assemble_to_bin_file
from ipu_apps.kernel_registry.registry import create_harness, kernel_spec


@dataclass(frozen=True)
class PreparedCase:
    params: Mapping[str, Any]
    bindings: Mapping[str, Any]
    check: Callable[[], None]


@dataclass(frozen=True)
class KernelCase:
    """Prepare files in a workspace, then check output after a completed run.

    Scalar defaults also describe command-line options. Preparation may validate
    richer options; no parser, emulator, or debugger belongs in a case factory.
    """
    prepare: Callable[..., PreparedCase]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    max_cycles: int = 1_000_000

    def __post_init__(self):
        reserved = {"kernel", "case", "list_cases", "max_cycles", "output", "help"}
        for name, default in self.defaults.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"invalid case option name: {name!r}")
            if name in reserved:
                raise ValueError(f"case option {name!r} conflicts with a runner option")
            if type(default) not in (str, int, float, bool):
                raise ValueError(f"case option {name!r} requires a str, int, float, or bool default")


def load_cases(kernel_name: str) -> Mapping[str, KernelCase]:
    spec = kernel_spec(kernel_name)
    module_name = spec.resource_package + ".cases"
    try:
        cases = import_module(module_name).CASES
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{kernel_name}: cannot load CASES from {module_name}: {exc}") from exc
    if (not isinstance(cases, Mapping) or "default" not in cases
            or not all(isinstance(n, str) and isinstance(c, KernelCase) for n, c in cases.items())):
        raise ValueError(f"{kernel_name}: CASES must declare a default KernelCase")
    return cases


def check_output_bytes(actual_path, expected_path):
    """Compare binary output without assertions that disappear under -O."""
    actual, expected = Path(actual_path).read_bytes(), Path(expected_path).read_bytes()
    if actual == expected:
        return
    if len(actual) != len(expected):
        raise ValueError(f"output size mismatch: got {len(actual)} bytes, expected {len(expected)}")
    for offset, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            raise ValueError(f"output mismatch at byte {offset}: got 0x{got:02x}, expected 0x{want:02x}")


def run_case(kernel_name: str, case: KernelCase, *, options=None, max_cycles=None,
             output_path=None, workspace=None, inst_path=None):
    """Assemble, prepare, construct, execute, and check a case.

    An optional workspace/instruction binary lets tests reuse assembly and
    inspect output files. Otherwise all temporary files are cleaned on every
    exit, including cancellation. Completed output is exported before checking
    so callers can inspect failed results. Returns the state and cycle count.
    """
    values = dict(case.defaults)
    overrides = dict(options or {})
    unknown = overrides.keys() - values.keys()
    if unknown:
        raise ValueError(f"unknown case options: {sorted(unknown)}")
    values.update(overrides)
    limit = case.max_cycles if max_cycles is None else max_cycles
    if limit <= 0:
        raise ValueError("max_cycles must be positive")
    if workspace is None:
        with TemporaryDirectory(prefix="ipu-case-") as tmp:
            return run_case(kernel_name, case, options=values, max_cycles=limit,
                            output_path=output_path, workspace=Path(tmp), inst_path=inst_path)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    prepared = case.prepare(workspace, **values)
    spec = kernel_spec(kernel_name)
    # Refuse unsupported cases before spending time assembling the kernel.
    spec.guard(**prepared.params)
    if inst_path is None:
        if not spec.asm:
            raise ValueError(f"{kernel_name}: SPEC must declare asm")
        source = files(spec.resource_package).joinpath(spec.asm)
        inst_path = workspace / "instructions.bin"
        assemble_to_bin_file(source.read_text(), str(inst_path))
    bindings = dict(prepared.bindings)
    if "inst_path" in bindings:
        raise ValueError("case bindings cannot replace the assembled instruction file")
    bindings["inst_path"] = inst_path
    app = create_harness(kernel_name, params=prepared.params, bindings=bindings)
    state, cycles = app.run(max_cycles=limit)
    if not state.is_halted:
        raise RuntimeError(f"{kernel_name} did not complete within {limit} cycles")
    if output_path is not None:
        source = bindings.get("output_path")
        if source is None:
            raise ValueError("case has no output file to export")
        Path(output_path).write_bytes(Path(source).read_bytes())
    try:
        prepared.check()
    except AssertionError as exc:
        raise AssertionError(f"{kernel_name}: {str(exc) or 'case output check failed'}") from exc
    return state, cycles
