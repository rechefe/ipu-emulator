"""Chain kernels in memory: one kernel's output region is the next one's input
(E06-15).

A :class:`Chain` runs kernels one after another on one reference model, so a
step reads what the step before it left in XMEM, with no file between them
and no harness internals patched. A step's outputs come back as
:class:`Resident` tensors; passing one as the next step's input pins that
input's region at the resident rows (placement checks it against the step's
other regions) instead of packing and writing it. A resident's layout, its
pack transform and its stored format, must be what the consumer declares, or
the step fails before it runs.

A step places its own regions by first fit and knows nothing of an earlier
step's residents but its inputs: a resident the next step does not take may
be overwritten by it.

    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale, {"rows": 2, "s": 2.0}, {"x": x})
    second = chain.run(scale, {"rows": 2, "s": 3.0}, {"x": first.resident["y"]})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness._codec import Array
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, checked_input
from ipu_harness.manifest import CaseDecl, KernelManifest
from ipu_harness.packing import stored_format, stored_rows
from ipu_harness.params import CasePlan, plan_case
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, RunResult, SharedModel, run_kernel
from ipu_isa.formats import ElementFormat

__all__ = ["Chain", "ChainStep", "Resident"]

#: The case name a chain step reports.
_STEP_CASE = "chain"


@dataclass(frozen=True)
class Resident:
    """A tensor already in XMEM: where it is and how it is laid out.

    Attributes:
        base_row: Its first row.
        rows: The rows it occupies.
        pack: The pack transform it was written with (``None``: none).
        format: The element format of its stored elements (an accumulator
            output's is FP32 or INT32).
    """

    base_row: int
    rows: int
    pack: str | None
    format: ElementFormat


@dataclass(frozen=True)
class ChainStep:
    """One step of a chain.

    Attributes:
        result: The kernel's run, as ``run_kernel`` gives it.
        resident: This step's outputs, by name, ready to feed the next step.
    """

    result: RunResult
    resident: Mapping[str, Resident]


class Chain:
    """Kernels run in turn on one model.

    Args:
        fmt: The element format every step runs in.
        options: How each step runs (limits, debugger, profile). A trace is
            written per step, as ``<stem>.<i>.jsonl`` beside ``trace``; the
            outputs per step, in ``output_dir/<i>``.

    Raises:
        ConfigurationError: ``options`` gives ``inputs`` (a step's inputs are
            ``run``'s).
    """

    def __init__(self, fmt: ElementFormat, *, options: RunOptions | None = None) -> None:
        self.format = fmt
        self.options = options or RunOptions()
        if self.options.inputs:
            raise ConfigurationError("a chain step takes its inputs from Chain.run, not --input")
        self.model = ReferenceModel(fmt)
        self._cache = AssemblyCache()
        self._steps = 0

    def _step_options(self) -> RunOptions:
        """The options of the next step: its own trace and output files."""
        step, options = self._steps, self.options
        return replace(
            options,
            trace=None
            if options.trace is None
            else options.trace.with_name(f"{options.trace.stem}.{step}.jsonl"),
            output_dir=None if options.output_dir is None else options.output_dir / str(step),
        )

    def run(
        self,
        kernel: str | KernelManifest,
        params: Mapping[str, object],
        inputs: Mapping[str, Array | Resident],
        *,
        check: bool = False,
    ) -> ChainStep:
        """Run one kernel on the chain's model.

        Plan the case; place it, each ``Resident`` input pinned at its rows;
        pack and write only the array inputs; write the registers; assemble
        and run to ``HALT``.

        Args:
            kernel: A kernel, or its name in the kernel tree.
            params: Its parameter values.
            inputs: Each input, as values or as an earlier step's output.
            check: Hold the outputs to their goldens (the kernel's reference,
                over the inputs, resident ones read back); an intermediate step
                need not, and its result has no comparisons.

        Returns:
            The run, and its outputs as ``Resident`` tensors.

        Raises:
            ConfigurationError: An input is missing or names no input; an
                array input is not finite numbers of its shape; a
                ``Resident``'s pack, format or rows differ from what the kernel
                declares, or it moves an input ``kernel.toml`` pins to a row.
            PlanError: The case is wrong, or a ``Resident`` overlaps another
                region (placement).
        """
        manifest = default_registry().kernel(kernel) if isinstance(kernel, str) else kernel
        decl = CaseDecl(_STEP_CASE, MappingProxyType(dict(params)))
        plan = plan_case(manifest, params, self.format, case=decl)
        pinned: dict[str, int] = {}
        arrays: dict[str, Array] = {}
        for name, value in inputs.items():
            tensor = manifest.tensors.get(name)
            if tensor is None or tensor.role != "input":
                # No input of the kernel: the runner names every wrong input.
                if isinstance(value, Resident):
                    pinned[name] = value.base_row
                else:
                    arrays[name] = value
            elif isinstance(value, Resident):
                _check_layout(manifest, name, value, plan)
                pinned[name] = value.base_row
            else:
                arrays[name] = checked_input(plan, name, value)
        case = Case(
            name=_STEP_CASE,
            values=MappingProxyType(dict(params)),
            formats=(self.format,),
            inputs=MappingProxyType({}),
            expected=MappingProxyType({}),
            generator=None,
            tier="default",
            decl=decl,
        )
        shared = SharedModel(self.model, MappingProxyType(arrays), MappingProxyType(pinned), check)
        options = self._step_options()
        self._steps += 1
        result = run_kernel(manifest, case, self.format, options, cache=self._cache, shared=shared)
        return ChainStep(result, _resident(manifest, result, plan))


def _check_layout(manifest: KernelManifest, name: str, value: Resident, plan: CasePlan) -> None:
    tensor = manifest.tensors[name]
    want = (tensor.pack, stored_format(tensor, plan.format), stored_rows(tensor, plan))
    have = (value.pack, value.format, value.rows)
    if tensor.row is not None and value.base_row != tensor.row:
        raise ConfigurationError(
            f"{manifest.name}: input {name} is pinned to row {tensor.row} in kernel.toml; "
            f"the resident tensor is at row {value.base_row}"
        )
    if have != want:
        raise ConfigurationError(
            f"{manifest.name}: input {name} is laid out as pack={want[0]!r}, "
            f"format={want[1].value}, {want[2]} rows; the resident tensor is "
            f"pack={have[0]!r}, format={have[1].value}, {have[2]} rows"
        )


def _resident(
    manifest: KernelManifest, result: RunResult, plan: CasePlan
) -> Mapping[str, Resident]:
    """Each output of ``result``, where its run left it."""
    if result.placement is None or not result.halted:
        return MappingProxyType({})
    out = {}
    for tensor in manifest.tensors.values():
        if tensor.role == "output":
            region = result.placement.region(tensor.name)
            out[tensor.name] = Resident(
                region.base_row, region.rows, tensor.pack, stored_format(tensor, plan.format)
            )
    return MappingProxyType(out)
