"""Run one case in one format, from the manifest to a verdict (E06-11, Q80).

:func:`run_kernel` is the only execution path for a kernel in the harness.
It plans the case once, makes its inputs, places and packs its tensors,
assembles its program, programs the registers, runs the program on the
reference model to ``HALT``, unpacks the outputs and compares them with
their goldens. It drives the model through the stable reference-model API
(``ipu_emu.reference``, E04-30) only.

A run that faults (a hardware fault F-1 to F-9, the cycle limit F-9 among
them) returns a result with its :class:`RunFault`; a wrong kernel answer
returns failed comparisons. Neither raises. A bad request (a parameter, a
file, the manifest) raises ``ConfigurationError``; a program that does not
assemble raises the assembler's error; an instruction the model cannot run in
the format (``ACTIVATE.QUANTIZE`` in E4) raises ``EmulatorError``, and no
outputs are written then.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ipu_as.session import AssemblyResult
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunState
from ipu_emu.reference import RunResult as ModelRun
from ipu_emu.trace import EMULATOR_VERSION, TraceWriter
from ipu_harness._codec import Array, encode
from ipu_harness.abi import apply, check_cr_reads, register_writes
from ipu_harness.assembly import AssemblyCache, assemble
from ipu_harness.cases import Case, inputs_for
from ipu_harness.compare import Comparison, compare, expected_for, tier_for
from ipu_harness.debug import DebugLaunch, attach_alias_profile, debug_run, new_alias_profile
from ipu_harness.manifest import KernelManifest, TensorDecl
from ipu_harness.packing import pack, stored_format, stored_rows, unpack
from ipu_harness.params import CasePlan, plan_case
from ipu_harness.placement import Placement, place
from ipu_harness.reference import evaluate, load_reference
from ipu_harness.stages import StageOptions, run_stages
from ipu_isa.codec.word import word_bytes
from ipu_isa.faults import FaultId
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.trace_schema import SCHEMA_VERSION, isa_digest

if TYPE_CHECKING:
    from ipu_emu.alias_profile import AliasProfile

__all__ = ["RunFault", "RunOptions", "RunResult", "SharedModel", "run_kernel"]


@dataclass(frozen=True)
class RunOptions:
    """How to run a case, besides the case itself.

    Attributes:
        max_cycles: The cycle limit; ``None`` keeps the model's, 0 removes it.
        inputs: Input files that replace the case's inputs, by tensor name
            (``--input x=path.npy``).
        output_dir: Where to write ``<output>.npy`` for each output, failed
            runs included (``--output``).
        trace: Where to write the run's trace (``--trace``), JSON Lines in
            the E04 trace schema. Not with ``debug``.
        debug: Stop in the debugger (``--debug``, E06-12).
        alias_profile: Profile the ISA aliases of this run (``--profile-aliases``):
            the profile attaches to the one run the case makes.
    """

    max_cycles: int | None = None
    inputs: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    output_dir: Path | None = None
    trace: Path | None = None
    debug: DebugLaunch | None = None
    alias_profile: bool = False

    def __post_init__(self) -> None:
        if self.debug is not None and self.trace is not None:
            raise ConfigurationError("--trace and --debug do not combine")


@dataclass(frozen=True)
class SharedModel:
    """A run on a model an earlier run left its state in (E06-15's ``Chain``).

    Attributes:
        model: The model; it keeps its XMEM between runs.
        inputs: The input tensors to pack and write, by name, as the format
            holds them (``cases.checked_input``).
        pinned: Inputs already in XMEM, by name, at these base rows: placed
            there, not written. Every input is in ``inputs`` or here, not both.
        check: Hold the outputs to their goldens (an intermediate step of a
            chain need not; its result then has no comparisons).
    """

    model: ReferenceModel
    inputs: Mapping[str, Array]
    pinned: Mapping[str, int]
    check: bool = False


@dataclass(frozen=True)
class RunFault:
    """The hardware fault a run stopped on.

    Attributes:
        fault: Its ID (F-1 to F-9; F-9 is the cycle limit).
        pc: The program counter of the word that faulted.
        line: That word's line of the rendered program, if found.
    """

    fault: FaultId
    pc: int
    line: str | None

    def describe(self) -> str:
        """One line for a report: ``F-9 at PC 4: BLT.S LR1 CR4 row_loop ;;``."""
        where = f"{self.fault.label} at PC {self.pc}"
        return f"{where}: {self.line}" if self.line else where


@dataclass(frozen=True)
class RunResult:
    """One case run in one format.

    Attributes:
        kernel: The kernel's name.
        case: The case's name (``sweep[rows=7]`` for a sweep point).
        format: The element format.
        cycles: The words the program retired.
        mac_utilization: Effective MAC utilization (0 to 1): multiplier lanes
            that did a multiply other than by a constant 1, over the run's
            peak.
        outputs: Each output's values, unpacked.
        raw_outputs: Each output region's XMEM bytes, before unpacking.
        comparisons: Each output held to its golden; empty without one.
        fault: The fault the run stopped on, if it faulted.
        alias_profile: The run's ``AliasProfile``, when it was profiled.
        quit: The user quit the debugger: the run has no result (no outputs,
            no comparisons) and did not pass.
        halted: The run reached ``HALT``.
        stages: A multi-stage kernel's stages (E06-14), each as its program
            and the words it retired; ``cycles`` is their sum. Empty for a
            one-program kernel.
        placement: Where the case's tensors were in XMEM (E06-15 chains an
            output region into the next kernel's input).
    """

    kernel: str
    case: str
    format: ElementFormat
    cycles: int
    mac_utilization: float
    outputs: Mapping[str, Array]
    raw_outputs: Mapping[str, bytes]
    comparisons: tuple[Comparison, ...]
    fault: RunFault | None
    alias_profile: AliasProfile | None = None
    quit: bool = False
    halted: bool = True
    stages: tuple[tuple[str, int], ...] = ()
    placement: Placement | None = None

    @property
    def passed(self) -> bool:
        """Whether the run reached ``HALT`` and every comparison passed."""
        return self.halted and self.fault is None and all(c.passed for c in self.comparisons)

    def describe(self) -> str:
        """The report line: ``scale/one_row/e4: PASS  cycles=9  MAC util=12.5%  y: ...``."""
        if self.quit:
            return f"{self.kernel}/{self.case}/{self.format.value}: QUIT  (no result)"
        verdict = "PASS" if self.passed else "FAIL"
        head = (
            f"{self.kernel}/{self.case}/{self.format.value}: {verdict}  cycles={self.cycles}  "
            f"MAC util={self.mac_utilization * 100:.1f}%"
        )
        parts = [head]
        if self.stages:
            parts.append("stages " + " ".join(f"{p}={c}" for p, c in self.stages))
        if self.fault is not None:
            parts.append(f"fault {self.fault.describe()}")
        parts.extend(
            f"{c.output}: max|err|={c.max_abs:g} ({c.tier.name})" if c.passed else c.describe()
            for c in self.comparisons
        )
        return "  ".join(parts)


def _row_bytes(plan: CasePlan) -> int:
    return FORMATS[plan.format].row_bytes


def _write_region(
    model: ReferenceModel, plan: CasePlan, placement: Placement, tensor: TensorDecl, data: bytes
) -> None:
    model.write_xmem(placement.region(tensor.name).base_row * _row_bytes(plan), data)


def _read_region(
    model: ReferenceModel, plan: CasePlan, placement: Placement, tensor: TensorDecl
) -> bytes:
    region = placement.region(tensor.name)
    return model.read_xmem(region.base_row * _row_bytes(plan), region.rows * _row_bytes(plan))


def _scratch_fill(tensor: TensorDecl, plan: CasePlan, placement: Placement) -> bytes:
    """A scratch region's starting bytes: its ``fill`` in every element of every
    row it occupies, padding lanes included (not through the pack transform,
    which pads with zeros)."""
    fmt = stored_format(tensor, plan.format)
    region_bytes = placement.region(tensor.name).rows * _row_bytes(plan)
    elements = region_bytes // FORMATS[fmt].element_bytes
    return encode(np.full(elements, tensor.fill, dtype=np.float64), fmt)


#: A comment runs from ``#`` or ``//`` to the end of the line.
_COMMENT = re.compile(r"#|//")
#: A label at the start of a word: ``row_loop:``.
_LABEL = re.compile(r"^\s*[A-Za-z_][\w.]*\s*:")


def _source_line(rendered: str, pc: int) -> str | None:
    """The rendered program's line where word ``pc`` starts, stripped.

    A word ends at ``;;`` (the last word may omit it) and may span lines; its
    first line with an instruction on it names it.
    """
    lines = rendered.splitlines()
    word, first = 0, None
    for number, line in enumerate(lines):
        code = _COMMENT.split(line, maxsplit=1)[0]
        for index, part in enumerate(code.split(";;")):
            if index:  # a ";;" ended the word
                if word == pc:
                    return lines[first if first is not None else number].strip()
                word, first = word + 1, None
            if first is None and _LABEL.sub("", part).strip():
                first = number
    if word == pc and first is not None:
        return lines[first].strip()
    return None


def _trace_header(words: tuple[int, ...], fmt: ElementFormat) -> dict[str, str]:
    """The trace header (``trace.v1.json``) of ``words`` run in ``fmt``."""
    width = word_bytes()
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.to_bytes(width, "little"))
    return {
        "schema_version": SCHEMA_VERSION,
        "isa_digest": isa_digest(),
        "format": fmt.value,
        "program_digest": digest.hexdigest(),
        "emulator_version": EMULATOR_VERSION,
    }


def _goldens(case: Case, plan: CasePlan, inputs: Mapping[str, Array]) -> dict[str, Array]:
    """Each output's expected values. The reference (and torch) loads only when
    an output has no expected file."""
    outputs = [t.name for t in plan.manifest.tensors.values() if t.role == "output"]
    ref_out = None
    if any(name not in case.expected for name in outputs):
        fn = load_reference(plan.manifest)
        if fn is not None:
            ref_out = evaluate(fn, inputs, plan.params)
    return expected_for(case, plan, ref_out)


def _execute(
    model: ReferenceModel,
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    words: tuple[int, ...],
    options: RunOptions,
) -> tuple[ModelRun | None, AliasProfile | None]:
    """Run the loaded model as ``options`` say: profiled, in the debugger, traced.

    Returns:
        The model's result (``None`` when the user quit the debugger), and the
        profile when the run was profiled.
    """
    profile = None
    if options.alias_profile:
        profile = new_alias_profile(kernel=manifest.name, case=case.name, format=fmt.value)
        attach_alias_profile(model, profile)
    if options.debug is not None:
        outcome = debug_run(
            model,
            options.debug.hook(manifest.name),
            break_on_entry=options.debug.break_on_entry,
            max_cycles=options.max_cycles,
        )
        return outcome.run, profile
    if options.trace is not None:
        with TraceWriter(options.trace, _trace_header(words, fmt)) as trace:
            return model.run(options.max_cycles, trace=trace), profile
    return model.run(options.max_cycles), profile


#: How a run went: the model's (last) result, ``None`` when the user quit
#: the debugger; the profile, when profiled; the words retired; the effective
#: MAC utilization; the rendered program of the (last) stage; and each
#: stage's program and cycles (empty for one program).
_Outcome = tuple[
    "ModelRun | None", "AliasProfile | None", int, float, str, tuple[tuple[str, int], ...]
]


def _execute_program(
    model: ReferenceModel,
    plan: CasePlan,
    placement: Placement,
    case: Case,
    options: RunOptions,
    program: AssemblyResult,
) -> _Outcome:
    """Run the kernel's one program, loaded, after writing its registers."""
    apply(model, register_writes(plan, placement))
    manifest, fmt = plan.manifest, plan.format
    run, profile = _execute(model, manifest, case, fmt, program.words, options)
    cycles = run.cycles if run is not None else model.stats().cycles
    mac = model.stats().effective_mac_utilization
    return run, profile, cycles, mac, program.rendered, ()


def _execute_stages(
    model: ReferenceModel,
    plan: CasePlan,
    placement: Placement,
    case: Case,
    options: RunOptions,
    cache: AssemblyCache | None,
) -> _Outcome:
    """Run a multi-stage kernel's hook (E06-14) as ``options`` say: one
    profile over every stage, each stage in the debugger, a trace per stage."""
    manifest = plan.manifest
    profile = None
    if options.alias_profile:
        profile = new_alias_profile(kernel=manifest.name, case=case.name, format=plan.format.value)
        attach_alias_profile(model, profile)
    debug = options.debug
    staged = run_stages(
        plan,
        placement,
        model,
        cache=cache,
        options=StageOptions(
            max_cycles=options.max_cycles,
            debug=debug.hook(manifest.name) if debug is not None else None,
            break_on_entry=debug.break_on_entry if debug is not None else False,
            trace=options.trace,
        ),
    )
    stages = tuple((stage.program, stage.cycles) for stage in staged.stages)
    rendered = staged.stages[-1].rendered
    return staged.result, profile, staged.cycles, staged.mac_utilization, rendered, stages


def _shared_inputs(plan: CasePlan, shared: SharedModel) -> Mapping[str, Array]:
    """``shared``'s array inputs, once its inputs are checked to be the kernel's."""
    manifest = plan.manifest
    declared = {t.name for t in manifest.tensors.values() if t.role == "input"}
    given = [*shared.inputs, *shared.pinned]
    if sorted(given) != sorted(declared):
        raise ConfigurationError(
            f"{manifest.name} takes inputs {sorted(declared)}, given {sorted(given)}"
        )
    return shared.inputs


def _write_tensors(
    model: ReferenceModel,
    plan: CasePlan,
    placement: Placement,
    inputs: Mapping[str, Array],
    shared: SharedModel | None,
) -> Mapping[str, Array]:
    """Write every input and scratch region but the pinned inputs, which are
    already in XMEM; return every input's values, the pinned ones read back
    (for the goldens)."""
    inputs = dict(inputs)
    for tensor in plan.manifest.tensors.values():
        if tensor.role == "input" and shared is not None and tensor.name in shared.pinned:
            data = _read_region(model, plan, placement, tensor)
            inputs[tensor.name] = unpack(tensor, data, plan)
        elif tensor.role == "input":
            _write_region(model, plan, placement, tensor, pack(tensor, inputs[tensor.name], plan))
        elif tensor.role == "scratch":
            _write_region(model, plan, placement, tensor, _scratch_fill(tensor, plan, placement))
    return inputs


def run_kernel(
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    options: RunOptions | None = None,
    *,
    cache: AssemblyCache | None = None,
    shared: SharedModel | None = None,
) -> RunResult:
    """Run ``case`` in ``fmt`` and hold its outputs to their goldens.

    Args:
        manifest: The kernel.
        case: One of its cases (``cases.expand``).
        fmt: The element format.
        options: Limits, input overrides, output and trace files.
        cache: Reuses programs already assembled (a sweep renders few).
        shared: Run on this model, with these inputs, some already in XMEM
            (E06-15); ``None``: a fresh model and the case's inputs.

    Returns:
        The result; ``passed`` says whether the run reached ``HALT`` and
        matched every golden.

    Raises:
        ConfigurationError: The case, its files or the program are wrong for
            the kernel (``PlanError`` locates it in ``kernel.toml``), or
            ``shared``'s inputs are not the kernel's.
        ipu_as.errors.AssemblerError: The program does not render, parse or
            encode.
        EmulatorError: An instruction the model cannot run in ``fmt``.
        OSError: An input, output or trace file cannot be read or written.
    """
    options = options or RunOptions()
    plan = plan_case(manifest, case.values, fmt, case=case.decl)
    if shared is None:
        inputs = inputs_for(case, plan, overrides=options.inputs).tensors
    else:
        inputs = _shared_inputs(plan, shared)
    tensors = plan.manifest.tensors
    pinned = shared.pinned if shared is not None else {}
    placement = place(
        plan, {name: stored_rows(t, plan) for name, t in tensors.items()}, pins=pinned
    )
    # A multi-stage kernel's hook assembles and loads each stage (E06-14).
    program = None if manifest.stages is not None else assemble(plan, cache)
    model = shared.model if shared is not None else ReferenceModel(fmt)
    if program is not None:
        check_cr_reads(manifest, program.rendered)
        model.load_program(program.words)
    inputs = _write_tensors(model, plan, placement, inputs, shared)
    if program is None:
        outcome = _execute_stages(model, plan, placement, case, options, cache)
    else:
        outcome = _execute_program(model, plan, placement, case, options, program)
    run, profile, cycles, mac_utilization, rendered, stage_cycles = outcome
    if run is None:  # the user quit the debugger: no result
        return RunResult(
            kernel=manifest.name,
            case=case.name,
            format=fmt,
            cycles=cycles,
            mac_utilization=mac_utilization,
            outputs=MappingProxyType({}),
            raw_outputs=MappingProxyType({}),
            comparisons=(),
            fault=None,
            alias_profile=profile,
            quit=True,
            halted=False,
            stages=stage_cycles,
            placement=placement,
        )

    raw = {
        t.name: _read_region(model, plan, placement, t)
        for t in tensors.values()
        if t.role == "output"
    }
    outputs = {name: unpack(tensors[name], data, plan) for name, data in raw.items()}
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        for name, values in outputs.items():
            np.save(options.output_dir / f"{name}.npy", values)

    fault = None
    comparisons: tuple[Comparison, ...] = ()
    if run.status.state is RunState.FAULTED:
        assert run.status.fault is not None  # a faulted status names its fault
        fault = RunFault(run.status.fault, run.status.pc, _source_line(rendered, run.status.pc))
    elif shared is None or shared.check:
        golden = _goldens(case, plan, inputs)
        comparisons = tuple(
            compare(
                name,
                outputs[name],
                want,
                tier_for(plan, tensors[name]),
                stored_format(tensors[name], fmt),  # an accumulator output is FP32 or INT32
            )
            for name, want in golden.items()
        )
    return RunResult(
        kernel=manifest.name,
        case=case.name,
        format=fmt,
        cycles=cycles,
        mac_utilization=mac_utilization,
        outputs=MappingProxyType(outputs),
        raw_outputs=MappingProxyType(raw),
        comparisons=comparisons,
        fault=fault,
        alias_profile=profile,
        halted=run.status.state is RunState.HALTED,
        stages=stage_cycles,
        placement=placement,
    )
