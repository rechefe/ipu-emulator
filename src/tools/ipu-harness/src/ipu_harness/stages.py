"""Multi-stage kernels: several programs on one machine state (E06-14, Q80).

A kernel whose ``kernel.toml`` names ``[kernel] stages = "stages.py:run"``
describes its stages in a few lines of Python instead of overriding the run.
The runner (:func:`ipu_harness.runner.run_kernel`) plans, places and packs
the case once, writes its inputs and scratch regions, then calls the hook
with a :class:`StageContext`. Each :meth:`StageContext.run_stage` assembles
one program, writes its registers from ``[abi]``-style names and runs it to
``HALT`` on the same model, so a stage reads what the one before it left in
XMEM. The runner then reads the outputs and compares them, as for a
one-program kernel.

The hook never computes an address or a register number: intermediate
regions are ``[scratch.*]`` tensors (placement owns their rows), and a stage
names the registers it reads as ``[abi]`` does. A stage may run another
kernel's program only from the same family folder (``FAMILY_DEPS``, E06-18,
will widen that).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from ipu_as.session import AssemblySession
from ipu_as.template import REQUIRED, Param, ParamSpec
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunResult, RunState, TraceWriter
from ipu_harness._plugins import load_file
from ipu_harness.abi import apply, check_cr_reads, register_writes, template_names
from ipu_harness.assembly import AssemblyCache, declared_params, search_path, template_values
from ipu_harness.debug import DebugHook, debug_run
from ipu_harness.manifest import KernelManifest
from ipu_harness.params import CasePlan
from ipu_harness.placement import Placement
from ipu_isa.constants import DStructureConfig, encode_dstructure
from ipu_isa.geometry import LANES

__all__ = ["StageContext", "StageOptions", "StageRun", "StagesRun", "run_stages"]


@dataclass(frozen=True)
class StageOptions:
    """How every stage runs.

    Attributes:
        max_cycles: Each stage's cycle limit unless the stage gives one;
            ``None`` keeps the model's.
        debug: The debugger hook every stage runs under (E06-12's
            ``DebugHook``), or ``None``.
        break_on_entry: With ``debug``, stop each stage before its first word.
        trace: Where each stage writes its trace, as ``<stem>.<i>.jsonl``
            beside this path (one trace per program), or ``None``.
    """

    max_cycles: int | None = None
    debug: DebugHook | None = None
    break_on_entry: bool = False
    trace: Path | None = None


@dataclass(frozen=True)
class StageRun:
    """One stage that ran.

    Attributes:
        program: The program, as the hook named it.
        cycles: The words it retired.
        mult_lane_ops: Its multiplier lanes that carried data.
        mult_identity_lane_ops: The part of ``mult_lane_ops`` that multiplied
            by a constant 1.
        rendered: The program's rendered text, for a fault report.
    """

    program: str
    cycles: int
    mult_lane_ops: int
    mult_identity_lane_ops: int
    rendered: str


@dataclass(frozen=True)
class StagesRun:
    """Every stage of one run.

    Attributes:
        stages: The stages that ran, in order; the last one faulted when
            ``result`` is a fault.
        result: The last stage's run (its status ends the kernel's run);
            ``None`` when the user quit the debugger in a stage.
    """

    stages: tuple[StageRun, ...]
    result: RunResult | None

    @property
    def cycles(self) -> int:
        """The words every stage retired."""
        return sum(stage.cycles for stage in self.stages)

    @property
    def mac_utilization(self) -> float:
        """Effective MAC utilization over every stage: lanes that multiplied
        by something other than a constant 1, over the stages' peak."""
        if self.cycles == 0:
            return 0.0
        useful = sum(s.mult_lane_ops - s.mult_identity_lane_ops for s in self.stages)
        return useful / (self.cycles * LANES)


class StageContext(Protocol):
    """What a kernel's stages hook gets.

    Attributes:
        plan: The checked case.
        placement: Its regions, scratch included.
    """

    plan: CasePlan
    placement: Placement

    def run_stage(
        self,
        program: str,
        *,
        params: Mapping[str, object] = MappingProxyType({}),
        abi: Mapping[str, str] | None = None,
        dstructure: DStructureConfig | None = None,
        max_cycles: int | None = None,
    ) -> int:
        """Run one program to ``HALT`` on the model the stages share.

        Args:
            program: ``"name.asm"`` in the kernel's folder, or
                ``"../other/name.asm"``, another kernel's in the same family.
            params: Template values besides the case's parameters, ``format``
                and ``shapes`` (``elements_in_row``).
            abi: Names to registers, as ``[abi]`` binds them
                (``{"x": "CR2"}``): each name is a tensor (its region base),
                a parameter or a derived value. ``None``: the kernel's
                ``[abi]``.
            dstructure: ``CR15``'s dstructure, unless ``abi`` binds ``CR15``;
                ``None``: the default (128 valid elements).
            max_cycles: The stage's cycle limit; ``None``: the run's.

        Returns:
            The words the stage retired.

        Raises:
            ConfigurationError: The program is outside the kernel's family, a
                name is not the case's, or the program reads an unbound CR.
        """
        ...


class _StageStop(Exception):
    """A stage faulted, or the user quit the debugger in it: the stages after
    it do not run."""


@dataclass
class _Stages:
    manifest: KernelManifest
    plan: CasePlan
    placement: Placement
    model: ReferenceModel
    cache: AssemblyCache
    options: StageOptions
    runs: list[StageRun] = field(default_factory=list)
    last: RunResult | None = None
    quit: bool = False

    def _program(self, program: str) -> Path:
        directory = self.manifest.directory.resolve()
        path = (directory / program).resolve()
        family = directory.parent
        if family not in path.parents:
            raise ConfigurationError(
                f"{self.manifest.name}: stage program {program!r} is outside the "
                f"kernel's family ({family.name}); FAMILY_DEPS (E06-18) is not there yet"
            )
        if not path.is_file():
            raise ConfigurationError(f"{self.manifest.name}: stage program {program!r} not found")
        return path

    def run_stage(
        self,
        program: str,
        *,
        params: Mapping[str, object] = MappingProxyType({}),
        abi: Mapping[str, str] | None = None,
        dstructure: DStructureConfig | None = None,
        max_cycles: int | None = None,
    ) -> int:
        path = self._program(program)
        bindings = self.manifest.abi if abi is None else MappingProxyType(dict(abi))
        clash = set(params) & set(template_values(self.plan))
        if clash:
            raise ConfigurationError(
                f"{self.manifest.name}: stage params {sorted(clash)} are already the case's"
            )
        declared = ParamSpec(
            (
                *declared_params(self.plan).params,
                *(Param(n, type(v), REQUIRED) for n, v in params.items()),
            )
        )
        session = AssemblySession(
            path.read_text(encoding="utf-8"),
            defines={**template_values(self.plan), **params},
            params=declared,
            search_path=(path.parent, *search_path(self.plan)),
            abi=template_names(self.manifest, bindings),
            name=path.name,
        )
        assembled = self.cache.get(session)
        check_cr_reads(self.manifest, assembled.rendered, abi=bindings, name=path.name)

        model = self.model
        model.load_program(assembled.words)
        writes = register_writes(self.plan, self.placement, abi=bindings)
        if dstructure is not None and "CR15" not in bindings.values():
            bits = encode_dstructure(
                valid_elements=dstructure.valid_elements,
                partition=dstructure.partition,
                pad_mode=dstructure.pad_mode,
            )
            writes = tuple(w for w in writes if w.register != "CR15")
            model.write_register("cr", bits, 15)
        apply(model, writes)

        options = self.options
        limit = options.max_cycles if max_cycles is None else max_cycles
        result: RunResult | None
        if options.debug is not None:
            outcome = debug_run(
                model, options.debug, break_on_entry=options.break_on_entry, max_cycles=limit
            )
            result, self.quit = outcome.run, outcome.quit
        elif options.trace is not None:
            stage_trace = options.trace.with_name(f"{options.trace.stem}.{len(self.runs)}.jsonl")
            with TraceWriter(stage_trace, model.trace_header()) as trace:
                result = model.run(limit, trace=trace)
        else:
            result = model.run(limit)
        stats = model.stats()
        self.runs.append(
            StageRun(
                program,
                stats.cycles,
                stats.mult_lane_ops,
                stats.mult_identity_lane_ops,
                assembled.rendered,
            )
        )
        self.last = result
        if result is None or result.status.state is not RunState.HALTED:
            raise _StageStop
        return result.cycles


def load_hook(manifest: KernelManifest) -> Callable[[StageContext], None]:
    """The kernel's stages hook (``[kernel] stages = "stages.py:run"``).

    Args:
        manifest: A kernel with ``stages``.

    Returns:
        The hook function.

    Raises:
        ConfigurationError: The kernel has no stages hook, or the file has no
            such function.
    """
    if manifest.stages is None:
        raise ConfigurationError(f"{manifest.name}: no [kernel] stages hook")
    file, _, name = manifest.stages.partition(":")
    hook = getattr(load_file(manifest.directory / file), name, None)
    if not callable(hook):
        raise ConfigurationError(f"{manifest.name}: {manifest.stages} is no function")
    return hook  # type: ignore[no-any-return]


def run_stages(
    plan: CasePlan,
    placement: Placement,
    model: ReferenceModel,
    *,
    cache: AssemblyCache | None = None,
    options: StageOptions | None = None,
) -> StagesRun:
    """Run the kernel's stages hook on ``model``, whose inputs and scratch
    regions the caller has written.

    Args:
        plan: The checked case of a kernel with a ``stages`` hook.
        placement: Its regions.
        model: The model every stage runs on.
        cache: Reuses programs already assembled.
        options: Limits, debugger and trace for every stage.

    Returns:
        The stages that ran; the run stops at a stage that faults or that
        the user quits in.

    Raises:
        ConfigurationError: The hook ran no stage, or a stage's request is
            wrong (see :meth:`StageContext.run_stage`).
    """
    manifest = plan.manifest
    stages = _Stages(
        manifest, plan, placement, model, cache or AssemblyCache(), options or StageOptions()
    )
    try:
        load_hook(manifest)(stages)
    except _StageStop:
        pass
    if not stages.runs:
        raise ConfigurationError(f"{manifest.name}: the stages hook ran no stage")
    return StagesRun(tuple(stages.runs), stages.last)
