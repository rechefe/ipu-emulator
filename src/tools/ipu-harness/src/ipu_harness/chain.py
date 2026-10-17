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

    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale, {"rows": 2, "s": 2.0}, {"x": x})
    second = chain.run(scale, {"rows": 2, "s": 3.0}, {"x": first.resident["y"]})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case
from ipu_harness.manifest import CaseDecl, KernelManifest
from ipu_harness.packing import stored_format, stored_rows
from ipu_harness.params import CasePlan, plan_case
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, RunResult, SharedModel, run_kernel
from ipu_isa.formats import ElementFormat

__all__ = ["Chain", "ChainStep", "Resident"]

#: The case name a chain step reports.
STEP_CASE = "chain"


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
        options: How each step runs (limits, debugger, profile, trace).
    """

    def __init__(self, fmt: ElementFormat, *, options: RunOptions | None = None) -> None:
        self.format = fmt
        self.options = options or RunOptions()
        self.model = ReferenceModel(fmt)
        self._cache = AssemblyCache()

    def run(
        self,
        kernel: str | KernelManifest,
        params: Mapping[str, object],
        inputs: Mapping[str, np.ndarray | Resident],
        *,
        compare: bool = False,
    ) -> ChainStep:
        """Run one kernel on the chain's model.

        Plan the case; place it, each ``Resident`` input pinned at its rows;
        pack and write only the array inputs; write the registers; assemble
        and run to ``HALT``.

        Args:
            kernel: A kernel, or its name in the kernel tree.
            params: Its parameter values.
            inputs: Each input, as values or as an earlier step's output.
            compare: Hold the outputs to their goldens (the kernel's reference,
                over the inputs, resident ones read back); an intermediate step
                need not.

        Returns:
            The run, and its outputs as ``Resident`` tensors.

        Raises:
            ConfigurationError: An input is missing or names no input; a
                ``Resident``'s pack, format or rows differ from what the kernel
                declares; or the model did not end the last step halted.
            PlanError: The case is wrong, or a ``Resident`` overlaps another
                region (placement).
        """
        manifest = default_registry().kernel(kernel) if isinstance(kernel, str) else kernel
        plan = plan_case(manifest, params, self.format)
        declared = {t.name for t in manifest.tensors.values() if t.role == "input"}
        if set(inputs) != declared:
            raise ConfigurationError(
                f"{manifest.name} takes inputs {sorted(declared)}, given {sorted(inputs)}"
            )
        pinned: dict[str, int] = {}
        arrays: dict[str, np.ndarray] = {}
        for name, value in inputs.items():
            if isinstance(value, Resident):
                _check_layout(manifest, name, value, plan)
                pinned[name] = value.base_row
            else:
                arrays[name] = value
        case = Case(
            name=STEP_CASE,
            values=MappingProxyType(dict(params)),
            formats=(self.format,),
            inputs=MappingProxyType({}),
            expected=MappingProxyType({}),
            generator=None,
            tier="default",
            decl=CaseDecl(STEP_CASE, MappingProxyType(dict(params))),
        )
        shared = SharedModel(
            self.model, MappingProxyType(arrays), MappingProxyType(pinned), compare
        )
        result = run_kernel(
            manifest, case, self.format, self.options, cache=self._cache, shared=shared
        )
        return ChainStep(result, _resident(manifest, result, plan))


def _check_layout(manifest: KernelManifest, name: str, value: Resident, plan: CasePlan) -> None:
    tensor = manifest.tensors[name]
    want = (tensor.pack, stored_format(tensor, plan.format), stored_rows(tensor, plan))
    have = (value.pack, value.format, value.rows)
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
