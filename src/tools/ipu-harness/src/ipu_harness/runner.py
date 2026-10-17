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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ipu_as.session import AssemblyResult
from ipu_emu.errors import ConfigurationError
from ipu_emu.ipu_state import DEFAULT_CYCLE_LIMIT
from ipu_emu.reference import ReferenceModel, RunState
from ipu_emu.reference import RunResult as ModelRun
from ipu_emu.trace import EMULATOR_VERSION, TraceWriter
from ipu_harness._codec import Array, encode
from ipu_harness.abi import RegisterWrite, apply, check_cr_reads, register_writes
from ipu_harness.assembly import AssemblyCache, assemble
from ipu_harness.cases import Case, inputs_for
from ipu_harness.compare import Comparison, compare, expected_for, tier_for
from ipu_harness.debug import DebugLaunch, attach_alias_profile, debug_run, new_alias_profile
from ipu_harness.manifest import KernelManifest, TensorDecl
from ipu_harness.packing import pack, stored_format, stored_rows, unpack
from ipu_harness.params import CasePlan, plan_case
from ipu_harness.placement import Placement, place
from ipu_harness.reference import evaluate, load_reference
from ipu_harness.stages import StageOptions, StagesRun, run_stages
from ipu_isa.codec.word import word_bytes
from ipu_isa.faults import FaultId
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.trace_schema import SCHEMA_VERSION, isa_digest

if TYPE_CHECKING:
    from ipu_emu.alias_profile import AliasProfile

__all__ = [
    "Executor",
    "RunContext",
    "RunFault",
    "RunOptions",
    "RunResult",
    "SharedModel",
    "Staged",
    "run_kernel",
    "stage",
]


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
        executor: Runs the loaded program in place of ``model.run``: a tool
            that steps the model itself (the Inspector's curated traces)
            gets the :class:`RunContext`. The runner then reads the status
            and the cycles from the model. Not with ``debug`` or ``trace``.
    """

    max_cycles: int | None = None
    inputs: Mapping[str, Path] = field(default_factory=lambda: MappingProxyType({}))
    output_dir: Path | None = None
    trace: Path | None = None
    debug: DebugLaunch | None = None
    alias_profile: bool = False
    executor: Executor | None = None

    def __post_init__(self) -> None:
        if self.debug is not None and self.trace is not None:
            raise ConfigurationError("--trace and --debug do not combine")
        if self.executor is not None and (self.debug is not None or self.trace is not None):
            raise ConfigurationError("an executor runs the program itself: no debug or trace")


@dataclass(frozen=True)
class RunContext:
    """A case loaded on the model and ready to run, for an :data:`Executor`.

    Attributes:
        model: The reference model: program loaded, inputs and scratch
            written, registers programmed, halted at PC 0.
        program: The assembled program (``words``, ``rendered``, ``symbols``).
        format: The element format (``plan.format``).
        plan: The checked case.
        placement: Where each tensor lives in XMEM.
        trace_header: The header of a trace of this run (``trace.v1.json``),
            for a tool that writes the records ``model.step()`` returns.
        cycle_limit: The run's cycle limit, resolved (0: none). ``step()`` has
            none of its own: an executor stops there, or calls ``model.run``.
    """

    model: ReferenceModel
    program: AssemblyResult
    format: ElementFormat
    plan: CasePlan
    placement: Placement
    trace_header: Mapping[str, str]
    cycle_limit: int


#: Runs a loaded case in place of ``model.run``: steps ``context.model`` to a
#: HALT, a fault or ``context.cycle_limit`` (``step()`` returns each word's commit
#: record). The runner then takes the status and cycles from the model, and
#: unpacks and compares as after any run; a run left RUNNING does not pass.
Executor = Callable[[RunContext], None]


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
    context: RunContext,
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    options: RunOptions,
) -> tuple[ModelRun | None, AliasProfile | None]:
    """Run the loaded model as ``options`` say: profiled, in the debugger, by the
    executor, traced.

    Returns:
        The model's result (``None`` when the user quit the debugger), and the
        profile when the run was profiled.
    """
    model = context.model
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
    if options.executor is not None:
        options.executor(context)
        return ModelRun(model.status, model.stats().cycles), profile
    if options.trace is not None:
        with TraceWriter(options.trace, dict(context.trace_header)) as trace:
            return model.run(options.max_cycles, trace=trace), profile
    return model.run(options.max_cycles), profile


def _execute_stages(
    model: ReferenceModel,
    plan: CasePlan,
    placement: Placement,
    case: Case,
    options: RunOptions,
    cache: AssemblyCache | None,
) -> tuple[ModelRun | None, AliasProfile | None, StagesRun]:
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
    return staged.result, profile, staged


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


@dataclass(frozen=True)
class Staged:
    """A case loaded into a model, every register written, not yet run.

    Attributes:
        plan: The checked case.
        inputs: Each input's values, rounded to the format.
        placement: Where each tensor sits in XMEM.
        program: The assembled program, loaded into ``model``.
        writes: The register writes applied (``[abi]`` and the dstructure).
        model: The model, halted at PC 0, its XMEM and registers set up.
    """

    plan: CasePlan
    inputs: Mapping[str, Array]
    placement: Placement
    program: AssemblyResult
    writes: tuple[RegisterWrite, ...]
    model: ReferenceModel

    def region_bytes(self, tensor: str) -> tuple[int, int]:
        """The XMEM byte range ``(start, size)`` of ``tensor``'s region."""
        region = self.placement.region(tensor)
        row_bytes = _row_bytes(self.plan)
        return region.base_row * row_bytes, region.rows * row_bytes


def _prepare(
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    inputs: Mapping[str, Path] | None,
    shared: SharedModel | None,
) -> tuple[CasePlan, Mapping[str, Array], Placement]:
    """The checked case, its inputs (``shared``'s, when given) and its placement."""
    plan = plan_case(manifest, case.values, fmt, case=case.decl)
    if shared is None:
        values = inputs_for(case, plan, overrides=inputs or {}).tensors
    else:
        values = _shared_inputs(plan, shared)
    tensors = plan.manifest.tensors
    placement = place(
        plan,
        {name: stored_rows(t, plan) for name, t in tensors.items()},
        pins=None if shared is None else shared.pinned,
    )
    return plan, values, placement


def stage(
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    *,
    inputs: Mapping[str, Path] | None = None,
    cache: AssemblyCache | None = None,
    shared: SharedModel | None = None,
) -> Staged:
    """Everything :func:`run_kernel` does before it runs the program.

    It plans the case, makes its inputs, places and packs its tensors,
    assembles and loads the program, writes the input and scratch regions and
    programs the registers. ``compare_setup`` (E07-01) reads the result.

    Args:
        manifest: The kernel.
        case: One of its cases.
        fmt: The element format.
        inputs: Input files that replace the case's inputs, by tensor name.
        cache: Reuses programs already assembled.
        shared: Stage on this model, with these inputs, some already in XMEM
            (E06-15); ``None``: a fresh model and the case's inputs.

    Returns:
        The staged model and what went into it.

    Raises:
        ConfigurationError: As for :func:`run_kernel`; or the kernel has a
            stages hook (E06-14), so no one program to load.
        ipu_as.errors.AssemblerError: The program does not render, parse or
            encode.
    """
    if manifest.stages is not None:
        raise ConfigurationError(f"{manifest.name}: a multi-stage kernel has no one program")
    plan, values, placement = _prepare(manifest, case, fmt, inputs, shared)
    program = assemble(plan, cache)
    check_cr_reads(manifest, program.rendered)

    model = ReferenceModel(fmt) if shared is None else shared.model
    model.load_program(program.words)
    values = _write_tensors(model, plan, placement, values, shared)
    writes = tuple(register_writes(plan, placement))
    apply(model, writes)
    return Staged(plan, MappingProxyType(dict(values)), placement, program, writes, model)


@dataclass(frozen=True)
class _Ran:
    """A case that ran, one program or every stage: what the report needs."""

    plan: CasePlan
    inputs: Mapping[str, Array]
    placement: Placement
    model: ReferenceModel
    run: ModelRun | None
    profile: AliasProfile | None
    cycles: int
    mac_utilization: float
    rendered: str
    stages: tuple[tuple[str, int], ...] = ()


def _run_program(
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    options: RunOptions,
    cache: AssemblyCache | None,
    shared: SharedModel | None,
) -> _Ran:
    """Stage the kernel's one program and run it as ``options`` say."""
    staged = stage(manifest, case, fmt, inputs=options.inputs, cache=cache, shared=shared)
    model, program = staged.model, staged.program
    limit = DEFAULT_CYCLE_LIMIT if options.max_cycles is None else options.max_cycles
    header = _trace_header(program.words, fmt)
    context = RunContext(model, program, fmt, staged.plan, staged.placement, header, limit)
    run, profile = _execute(context, manifest, case, fmt, options)
    stats = model.stats()
    cycles = run.cycles if run is not None else stats.cycles
    return _Ran(
        staged.plan,
        staged.inputs,
        staged.placement,
        model,
        run,
        profile,
        cycles,
        stats.effective_mac_utilization,
        program.rendered,
    )


def _run_staged_kernel(
    manifest: KernelManifest,
    case: Case,
    fmt: ElementFormat,
    options: RunOptions,
    cache: AssemblyCache | None,
    shared: SharedModel | None,
) -> _Ran:
    """Write a multi-stage kernel's tensors and run its stages hook (E06-14)."""
    if options.executor is not None:
        raise ConfigurationError(f"{manifest.name}: an executor runs one program, not stages")
    plan, values, placement = _prepare(manifest, case, fmt, options.inputs, shared)
    model = ReferenceModel(fmt) if shared is None else shared.model
    values = _write_tensors(model, plan, placement, values, shared)
    run, profile, staged = _execute_stages(model, plan, placement, case, options, cache)
    return _Ran(
        plan,
        MappingProxyType(dict(values)),
        placement,
        model,
        run,
        profile,
        staged.cycles,
        staged.mac_utilization,
        staged.stages[-1].rendered,
        tuple((s.program, s.cycles) for s in staged.stages),
    )


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
    if manifest.stages is None:
        ran = _run_program(manifest, case, fmt, options, cache, shared)
    else:
        ran = _run_staged_kernel(manifest, case, fmt, options, cache, shared)
    plan, inputs, placement, model, run = ran.plan, ran.inputs, ran.placement, ran.model, ran.run
    tensors = plan.manifest.tensors
    if run is None:  # the user quit the debugger: no result
        return RunResult(
            kernel=manifest.name,
            case=case.name,
            format=fmt,
            cycles=ran.cycles,
            mac_utilization=ran.mac_utilization,
            outputs=MappingProxyType({}),
            raw_outputs=MappingProxyType({}),
            comparisons=(),
            fault=None,
            alias_profile=ran.profile,
            quit=True,
            halted=False,
            stages=ran.stages,
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
        fault = RunFault(run.status.fault, run.status.pc, _source_line(ran.rendered, run.status.pc))
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
        cycles=ran.cycles,
        mac_utilization=ran.mac_utilization,
        outputs=MappingProxyType(outputs),
        raw_outputs=MappingProxyType(raw),
        comparisons=comparisons,
        fault=fault,
        alias_profile=ran.profile,
        halted=run.status.state is RunState.HALTED,
        stages=ran.stages,
        placement=placement,
    )
