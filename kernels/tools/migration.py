"""What compare_setup and compare_legacy share: the old harness and the new one, side by side.

Temporary: the family migrations (E07-02 to E07-24) prove each kernel with
them, and E07-25 deletes this package with ``ipu_apps``.

The old side is an ``ipu_apps`` kernel, run through its own ``cases.py`` and
``create_harness``; the new side is a ``kernel.toml`` kernel, run through
``ipu_harness.runner``. A case of the old kernel is named as its baseline
record is (E02-07): a ``CASES`` name, or the option overrides of a sweep of the
default case (``rows=8,n=200``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import ipu_emu.emulator as emulator
from ipu_apps.kernel_registry import baseline
from ipu_apps.kernel_registry.cases import (
    KernelCase,
    assemble_kernel,
    load_cases,
    options_label,
    run_case,
)
from ipu_apps.kernel_registry.registry import create_harness, kernel_spec
from ipu_apps.kernel_registry.spec import KernelSpec
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_config import DEFAULT_DSTRUCTURE_CR
from ipu_emu.ipu_state import IpuState
from ipu_emu.reference import ReferenceModel
from ipu_harness.cases import Case, expand, inputs_for
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.params import plan_case
from ipu_harness.registry import Registry, kernel_tree
from ipu_harness.runner import SharedModel, Staged
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.registers import REGISTER_DEFINITIONS

#: The E02 asm digests (E02-01), repo-relative.
ASM_DIGESTS = "src/tools/ipu-apps/test/baselines/asm_digests.json"
#: Where the old kernels' packages sit, repo-relative.
LEGACY_SRC = "src/tools/ipu-apps/src"
#: How many differences a report lists before it only counts them.
SHOWN = 10


#: What every CR and LR holds before either side's setup, but the hardwired
#: CR0 and CR1 and CR15 (the runner writes its default dstructure; an old
#: harness relied on the reset one). A register the old harness wrote and
#: the new kernel's [abi] does not bind then differs (compare_setup), and a
#: register the program reads that neither side writes changes the output
#: (compare_legacy): a kernel run after another on one model inherits its
#: registers, so the reset values are no contract.
POISON = 0xA5A5

#: The registers that hold :data:`POISON` before setup, as ``(bank, index)``.
POISONED: tuple[tuple[str, int], ...] = tuple(
    (bank, index)
    for bank in ("cr", "lr")
    for index in range(REGISTER_DEFINITIONS[bank]["count"])
    if bank == "lr" or index not in (*CR_READ_ONLY_INITIAL_VALUES, DEFAULT_DSTRUCTURE_CR)
)


def poison_state(state: IpuState) -> None:
    """Write :data:`POISON` into an old harness's state's :data:`POISONED` registers."""
    for bank, index in POISONED:
        state.regfile.set_scalar(bank, index, POISON)


def poisoned_model(fmt: ElementFormat) -> ReferenceModel:
    """A fresh model whose :data:`POISONED` registers hold :data:`POISON`."""
    model = ReferenceModel(fmt)
    for bank, index in POISONED:
        model.write_register(bank, POISON, index)
    return model


def poisoned_shared(manifest: KernelManifest, case: Case, fmt: ElementFormat) -> SharedModel:
    """``case``'s inputs on a :func:`poisoned_model`, for ``stage`` and
    ``run_kernel`` (``shared=``), which then write only what the kernel binds."""
    plan = plan_case(manifest, case.values, fmt, case=case.decl)
    values = dict(inputs_for(case, plan).tensors)
    return SharedModel(poisoned_model(fmt), values, {}, check=True)


def runfile(path: str) -> Path:
    """A repo-relative file, from the runfiles (Bazel) or else the checkout."""
    for base in (os.environ.get("RUNFILES_DIR"), None):
        root = Path(base) / "_main" if base else Path.cwd()
        if (root / path).exists():
            return root / path
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace and (Path(workspace) / path).exists():
        return Path(workspace) / path
    raise FileNotFoundError(f"{path}: not in the runfiles or the checkout")


def kernels_root(given: Path | None) -> Path:
    """The kernel tree to read new kernels from: ``given``, else the checkout's
    ``kernels/`` under ``bazel run`` (so an edit shows without a rebuild), else
    the runfiles' tree."""
    if given is not None:
        return given
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        return Path(workspace) / "kernels"
    return kernel_tree()


def new_kernel(name_or_dir: str, root: Path | None = None) -> KernelManifest:
    """A new kernel: a folder holding ``kernel.toml``, or a kernel name in the tree."""
    base = Path(os.environ.get("BUILD_WORKING_DIRECTORY", "."))
    path = Path(name_or_dir)
    path = path if path.is_absolute() else base / path
    if (path / "kernel.toml").is_file():
        return load_manifest(path)
    return Registry.load(kernels_root(root)).kernel(name_or_dir)


def family_of(spec: KernelSpec) -> str:
    """An old kernel's family: ``reshape`` for ``ipu_apps.kernels.reshape.identity``."""
    return spec.resource_package.split(".")[2]


def legacy_asm_path(spec: KernelSpec) -> str:
    """An old kernel's program, repo-relative (its key in the E02 asm digests)."""
    return f"{LEGACY_SRC}/{spec.resource_package.replace('.', '/')}/{spec.asm}"


def e02_asm_digest(spec: KernelSpec) -> str | None:
    """The E02 ``.bin`` digest of the old kernel's program, if recorded."""
    digests = json.loads(runfile(ASM_DIGESTS).read_text(encoding="utf-8"))
    record = digests.get(legacy_asm_path(spec))
    return None if record is None else record["bin_sha256"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- cases -------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyCase:
    """One case of an old kernel: a ``CASES`` entry and its option overrides."""

    id: str
    name: str
    case: KernelCase
    options: Mapping[str, Any]

    @property
    def values(self) -> dict[str, Any]:
        return {**self.case.defaults, **self.options}


def _option(text: str, default: Any) -> Any:
    if isinstance(default, bool):
        return text == "True"
    return type(default)(text)


def legacy_case(kernel: str, case_id: str) -> LegacyCase:
    """The old kernel's case ``case_id``: a ``CASES`` name, or ``k=v,...``
    overrides of its default case (a ``case_tests`` sweep)."""
    cases = load_cases(kernel)
    if case_id in cases:
        return LegacyCase(case_id, case_id, cases[case_id], {})
    default = cases["default"]
    options = {}
    for part in case_id.split(","):
        key, sep, text = part.partition("=")
        if not sep or key not in default.defaults:
            raise ConfigurationError(
                f"{kernel}: no case {case_id!r} (cases: {', '.join(cases)}; "
                f"or overrides of the default case's {sorted(default.defaults)})"
            )
        options[key] = _option(text, default.defaults[key])
    if options_label(options) != case_id:
        raise ConfigurationError(f"{kernel}: {case_id!r} does not name the overrides it parses to")
    return LegacyCase(case_id, "default", default, options)


def legacy_case_ids(kernel: str) -> list[str]:
    """Every case of an old kernel that its E02 baseline records, in its
    ``CASES`` order, then the sweeps; ``CASES`` alone without a baseline."""
    names = list(load_cases(kernel))
    recorded = baseline.load(kernel)
    if not recorded:
        return names
    return [n for n in names if n in recorded] + sorted(set(recorded) - set(names))


def new_case(manifest: KernelManifest, case_id: str, mapping: Mapping[str, str]) -> Case:
    """The new kernel's case for the old case ``case_id``: ``mapping[case_id]``,
    else the case of that name, else the one sweep point named ``<case>[case_id]``."""
    cases = {case.name: case for case in expand(manifest)}
    wanted = mapping.get(case_id, case_id)
    if wanted in cases:
        return cases[wanted]
    points = [c for name, c in cases.items() if name.endswith(f"[{wanted}]")]
    if len(points) == 1:
        return points[0]
    raise ConfigurationError(
        f"{manifest.name}: no case for the old case {case_id!r} "
        f"(cases: {', '.join(cases)}; map one with --case-map {case_id}=NAME)"
    )


def case_format(state: IpuState, case: Case) -> ElementFormat:
    """The format both sides run in: the old harness's (FP32 for every legacy
    kernel but the FP8 cases of ``fully_connected``)."""
    if state.fmt not in case.formats:
        raise ConfigurationError(
            f"the old harness runs in {state.fmt.value}; the new case {case.name} "
            f"runs in {', '.join(f.value for f in case.formats)}"
        )
    return state.fmt


# --- the old harness ---------------------------------------------------------------


@dataclass
class LegacySetup:
    """An old harness's state right after ``setup()``, and what it wrote."""

    state: IpuState
    writes: list[tuple[int, int]] = field(default_factory=list)


def legacy_setup(kernel: str, case: LegacyCase, workspace: Path) -> LegacySetup:
    """Build the old harness through ``create_harness`` and run only its ``setup()``
    on a fresh state (the kernel's own execution config: FP32 for all but the FP8
    cases of ``fully_connected``), its :data:`POISONED` registers poisoned first."""
    spec = kernel_spec(kernel)
    workspace.mkdir(parents=True, exist_ok=True)
    prepared = case.case.prepare(workspace, **case.values)
    spec.guard(**prepared.params)
    bindings = {**prepared.bindings, "inst_path": assemble_kernel(kernel, workspace)}
    app = create_harness(kernel, params=prepared.params, bindings=bindings)
    state = app.make_state()
    poison_state(state)
    setup = LegacySetup(state)

    def observe(kind: str, address: int, size: int, data: object) -> None:
        if kind == "write" and size:
            setup.writes.append((address, size))

    remove = state.xmem.add_observer(observe)
    try:
        app.setup(state)
    finally:
        remove()
    return setup


@dataclass
class LegacyRun:
    """An old harness's whole run of one case."""

    state: IpuState
    cycles: int
    programs: list[bytes]
    record: baseline.CaseRecord


@contextmanager
def _programs_loaded() -> Iterator[list[bytes]]:
    """Each ``.bin`` the old harness loads while the block runs (a kernel whose
    program is rendered per run loads its own)."""
    loaded: list[bytes] = []
    original = emulator.load_program_from_binary

    def load(state: IpuState, path: str | Path) -> None:
        loaded.append(Path(path).read_bytes())
        original(state, path)

    emulator.load_program_from_binary = load
    try:
        yield loaded
    finally:
        emulator.load_program_from_binary = original


def legacy_run(kernel: str, case: LegacyCase, workspace: Path) -> LegacyRun:
    """Run the old harness's case to its end, its own check included, recording
    it as its E02 baseline does."""
    recorder = baseline.Recorder()
    with _programs_loaded() as programs:
        state, cycles = run_case(
            kernel,
            case.case,
            options=case.values,
            workspace=workspace,
            recorder=recorder,
        )
    assert recorder.record is not None  # run_case records every finished run
    return LegacyRun(state, cycles, programs, recorder.record)


# --- comparing a state with a staged one --------------------------------------------


def is_bound_reset(bank: str, index: int, old: int, new: int) -> bool:
    """Whether the new kernel binds a register the old setup left alone to the
    value the old harness relied on, 0 from reset: old still :data:`POISON`,
    new 0. That is no difference (the new kernel states what the old one
    assumed), so :func:`register_differences` reports it apart. "Left alone"
    is read from the value: an old setup that wrote :data:`POISON` itself
    would look the same (a value no kernel uses as a parameter)."""
    return (bank, index) in POISONED and old == POISON and new == 0


def register_differences(state: IpuState, staged: Staged) -> tuple[list[str], list[str]]:
    """Every CR and LR whose value differs, as ``CR7: old 14, new 15``, and
    apart from them each bound reset value (:func:`is_bound_reset`), as ``LR0``."""
    lines, resets = [], []
    for bank in ("cr", "lr"):
        raw = state.regfile.raw_readonly(bank)
        count = REGISTER_DEFINITIONS[bank]["count"]
        size = len(raw) // count
        for index in range(count):
            old = int.from_bytes(raw[index * size : (index + 1) * size], "little")
            new = staged.model.read_register(bank, index)
            assert isinstance(new, int)  # an LR or CR reads as an int
            if is_bound_reset(bank, index, old, new):
                resets.append(f"{bank.upper()}{index}")
            elif old != new:
                lines.append(f"{bank.upper()}{index}: old {old}, new {new}")
    return lines, resets


def _merged(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """``(start, size)`` ranges, overlapping and touching ones joined, sorted."""
    merged: list[list[int]] = []
    for start, size in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], start + size)
        else:
            merged.append([start, start + size])
    return [(start, end - start) for start, end in merged]


def written_ranges(setup: LegacySetup, staged: Staged) -> list[tuple[int, int]]:
    """The union of the XMEM the old ``setup()`` wrote and the new regions the
    runner writes (inputs and scratch)."""
    new = [
        staged.region_bytes(name)
        for name, tensor in staged.plan.manifest.tensors.items()
        if tensor.role in ("input", "scratch")
    ]
    return _merged(setup.writes + new)


def locate(staged: Staged, address: int) -> str:
    """Where a byte sits: ``x row 2 lane 5``, by the new placement; ``row R lane L``
    outside every region."""
    fmt = FORMATS[staged.plan.format]
    row_bytes, element = fmt.row_bytes, fmt.element_bytes
    for name in staged.plan.manifest.tensors:
        start, size = staged.region_bytes(name)
        if start <= address < start + size:
            offset = address - start
            return f"{name} row {offset // row_bytes} lane {offset % row_bytes // element}"
    return f"row {address // row_bytes} lane {address % row_bytes // element} (no region)"


def xmem_differences(
    state: IpuState, staged: Staged, ranges: list[tuple[int, int]]
) -> tuple[list[str], int]:
    """The elements of ``ranges`` whose bytes differ between the old state and
    the staged model.

    Returns:
        The first :data:`SHOWN` differences, each located by region, row and
        lane, and how many elements differ in all.
    """
    element = FORMATS[staged.plan.format].element_bytes
    shown: list[str] = []
    total = 0
    for start, size in ranges:
        old = np.frombuffer(state.xmem.snapshot_bytes(start, size), dtype=np.uint8)
        new = np.frombuffer(staged.model.read_xmem(start, size), dtype=np.uint8)
        # Each differing element by its address, aligned to the element.
        elements = np.unique((start + np.flatnonzero(old != new)) // element * element)
        total += len(elements)
        for address in elements[: max(0, SHOWN - len(shown))].tolist():
            lo, hi = max(address - start, 0), min(address - start + element, size)
            a, b = bytes(old[lo:hi]).hex(), bytes(new[lo:hi]).hex()
            shown.append(f"{locate(staged, address)} (byte {address}): old {a}, new {b}")
    return shown, total
