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

Every stage runs in the case's one format, on one model. ``max_cycles`` limits
each stage, not the run. Under a debugger no trace is written, as for a
one-program kernel.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from ipu_as.session import AssemblyResult
from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel, RunResult, RunState, Stats, TraceWriter
from ipu_harness._plugins import load_file
from ipu_harness.abi import apply, check_cr_reads, register_writes
from ipu_harness.assembly import AssemblyCache, assemble_program
from ipu_harness.debug import DebugHook, debug_run
from ipu_harness.manifest import KernelManifest, check_abi
from ipu_harness.params import CasePlan
from ipu_harness.placement import Placement, check_aaq_spill
from ipu_isa.constants import DStructureConfig
from ipu_isa.geometry import LANES

__all__ = [
    "StageContext",
    "StageOptions",
    "StageRun",
    "StagesRun",
    "load_hook",
    "run_stages",
    "stage_programs",
]


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
        stats: The model's statistics for the stage alone (``load_program``
            clears them), or ``None`` (a hand-made record).
    """

    program: str
    cycles: int
    mult_lane_ops: int
    mult_identity_lane_ops: int
    rendered: str
    stats: Stats | None = None


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

    @property
    def stats(self) -> Stats | None:
        """Every stage's statistics summed, the ratios over the sum (what the
        benchmark reports for the run, as the ipu_apps harness summed its
        stages'); ``None`` when a stage has none."""
        per = [s.stats for s in self.stages]
        if not per or any(st is None for st in per):
            return None
        counts = {
            name: sum(getattr(st, name) for st in per)
            for name in (
                "cycles",
                "mult_active_cycles",
                "acc_active_cycles",
                "xmem_reads",
                "xmem_writes",
                "mult_lane_ops",
                "mult_identity_lane_ops",
            )
        }
        peak = counts["cycles"] * LANES
        return Stats(
            **counts,
            lane_occupancy=counts["mult_lane_ops"] / peak if peak else 0.0,
            effective_mac_utilization=self.mac_utilization,
        )


class StageContext(Protocol):
    """What a kernel's stages hook gets (it reads ``plan`` and ``placement``)."""

    @property
    def plan(self) -> CasePlan:
        """The checked case."""
        ...

    @property
    def placement(self) -> Placement:
        """Its regions, scratch included."""
        ...

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
            params: Template values besides the case's parameters, derived
                values, ``format`` and ``shapes`` (``elements_in_row``).
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


class _StageStop(BaseException):
    """A stage faulted, or the user quit the debugger in it: the stages after
    it do not run. A ``BaseException``, so a hook's ``except Exception`` does
    not swallow it."""


def _program(manifest: KernelManifest, program: str) -> Path:
    # Lexical, not resolve(): under Bazel the files are symlinks out of
    # the runfiles tree (as kernel_test.py keeps them).
    directory = Path(os.path.abspath(manifest.directory))
    path = Path(os.path.normpath(directory / program))
    family = directory.parent
    if family not in path.parents:
        raise ConfigurationError(
            f"{manifest.name}: stage program {program!r} is outside the "
            f"kernel's family ({family.name}); FAMILY_DEPS (E06-18) is not there yet"
        )
    if not path.is_file():
        raise ConfigurationError(f"{manifest.name}: stage program {program!r} not found")
    return path


def _assemble_stage(
    plan: CasePlan,
    placement: Placement | None,
    cache: AssemblyCache,
    program: str,
    *,
    params: Mapping[str, object],
    abi: Mapping[str, str] | None,
) -> tuple[Mapping[str, str], AssemblyResult]:
    """One stage's register table and checked program (see ``run_stage``)."""
    manifest = plan.manifest
    path = _program(manifest, program)
    bindings = manifest.abi if abi is None else check_abi(manifest, abi, where=f"stage {program}")
    assembled = assemble_program(plan, path, cache, params=params, abi=bindings)
    check_cr_reads(manifest, assembled.rendered, abi=bindings, name=path.name)
    if placement is not None:
        check_aaq_spill(plan, placement, assembled.rendered)
    return bindings, assembled


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
    stopped: bool = False

    def run_stage(
        self,
        program: str,
        *,
        params: Mapping[str, object] = MappingProxyType({}),
        abi: Mapping[str, str] | None = None,
        dstructure: DStructureConfig | None = None,
        max_cycles: int | None = None,
    ) -> int:
        if self.stopped:
            raise _StageStop  # an earlier stage ended the run; the hook caught its stop
        bindings, assembled = _assemble_stage(
            self.plan, self.placement, self.cache, program, params=params, abi=abi
        )

        model = self.model
        model.load_program(assembled.words)
        # No table of its own: the kernel's [abi], offsets included (register_writes
        # applies a tensor's [abi] offset only to the kernel's own bindings).
        own = None if abi is None else bindings
        apply(model, register_writes(self.plan, self.placement, abi=own, dstructure=dstructure))

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
                stats,
            )
        )
        self.last = result
        if result is None or result.status.state is not RunState.HALTED:
            self.stopped = True
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


@dataclass
class _Assembling:
    """A stages context that assembles each stage and runs none."""

    plan: CasePlan
    regions: Placement | None
    cache: AssemblyCache
    programs: list[tuple[str, AssemblyResult]] = field(default_factory=list)

    @property
    def placement(self) -> Placement:
        """The case's regions, when the caller placed it."""
        if self.regions is None:
            raise ConfigurationError(
                f"{self.plan.manifest.name}: the stages hook read the placement of a "
                "case assembled without one"
            )
        return self.regions

    def run_stage(
        self,
        program: str,
        *,
        params: Mapping[str, object] = MappingProxyType({}),
        abi: Mapping[str, str] | None = None,
        dstructure: DStructureConfig | None = None,
        max_cycles: int | None = None,
    ) -> int:
        _, assembled = _assemble_stage(
            self.plan, self.regions, self.cache, program, params=params, abi=abi
        )
        self.programs.append((program, assembled))
        return 0


def stage_programs(
    plan: CasePlan, placement: Placement | None = None, *, cache: AssemblyCache | None = None
) -> tuple[tuple[str, AssemblyResult], ...]:
    """Every program the kernel's stages hook runs for ``plan``, assembled and
    checked as :meth:`StageContext.run_stage` does, but not run (each stage
    returns 0 cycles). For ``:assemble_<name>`` and the tree's program checks.

    Args:
        plan: The checked case of a kernel with a ``stages`` hook.
        placement: Its regions, for the ``ST.AAQ`` spill check; ``None``
            skips that check (the programs need no placement to assemble).
        cache: Reuses programs already assembled.

    Returns:
        Each stage's program, as the hook named it, and its assembly, in order.

    Raises:
        ConfigurationError: The hook ran no stage, or a stage's request is wrong.
    """
    context = _Assembling(plan, placement, cache or AssemblyCache())
    load_hook(plan.manifest)(context)
    if not context.programs:
        raise ConfigurationError(f"{plan.manifest.name}: the stages hook ran no stage")
    return tuple(context.programs)
