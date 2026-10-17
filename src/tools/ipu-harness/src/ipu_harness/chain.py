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

A step places its own regions by first fit, or where ``at`` puts them
(``chain.run(k, params, inputs, at={"y": 0})``: an output at row 0, for a
program that reads its base through ``CR0``), and knows nothing of an
earlier step's residents but its inputs, so a step may overwrite a resident it does
not take. The chain keeps track: a resident that a later step's placed regions
cover is stale, and passing it fails. The rows an ``ST.AAQ`` store spills past
an output or a scratch region (three in INT8, OQ-35) count as covered. A resident the chain
*holds* (:meth:`Chain.hold`, until :meth:`Chain.release`) is never covered:
every later step places around it, and a step that pins a region on it (or
spills onto it) fails before it runs. Hold a tensor a step several steps on
reads, such as a residual.

A tensor's rows can be used in parts: :meth:`Resident.slice` is a part of a
resident, an input of a step that reads only it (one third of a QKV output).
:meth:`Chain.reserve` holds free rows for steps to write into, a slice each
(``at={"y": whole.slice(g, 1)}``), such as one tensor many one-row steps
fill; write the slices in row order, as each ``ST.AAQ`` spill lands on the
next rows.

Only XMEM is the chain's state. A step's registers start as the step before
left them, as on the hardware: ``load_program`` resets the PC, and the step
writes only its ``[abi]`` registers and CR15; LRs, the vector registers, the
accumulator and the flags carry over. ``reset=("r_mask",)`` puts the named
register banks back to their reset values first, for a program that assumes
them (R_MASK all ones: every lane on) after one that left them changed.

    chain = Chain(ElementFormat.INT8)
    first = chain.run(scale, {"rows": 2, "s": 2.0}, {"x": x})
    second = chain.run(scale, {"rows": 2, "s": 3.0}, {"x": first.resident["y"]})
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ipu_emu.errors import ConfigurationError
from ipu_emu.reference import ReferenceModel
from ipu_harness._codec import Array
from ipu_harness.assembly import AssemblyCache, assemble
from ipu_harness.cases import Case, checked_input
from ipu_harness.manifest import CaseDecl, KernelManifest
from ipu_harness.packing import stored_format, stored_rows
from ipu_harness.params import CasePlan, PlanError, plan_case
from ipu_harness.placement import Region, aaq_spill_rows, place
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, RunResult, SharedModel, run_kernel
from ipu_isa.constants import CR_READ_ONLY_INITIAL_VALUES
from ipu_isa.formats import ElementFormat, addressable_rows
from ipu_isa.registers import REGISTER_DEFINITIONS

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
        step: The chain step that wrote it (0 for the first), so a later
            step's output in the same rows is a different tensor.
        parent: The resident this is a :meth:`slice` of (``None``: a whole
            tensor); it is in XMEM while its parent is.
    """

    base_row: int
    rows: int
    pack: str | None
    format: ElementFormat
    step: int = 0
    parent: Resident | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.base_row < 0 or self.rows < 1:
            raise ConfigurationError(
                f"a resident tensor starts at a row >= 0 and has rows >= 1, "
                f"not row {self.base_row}, {self.rows} rows"
            )

    def overlaps(self, region: Region) -> bool:
        """Whether ``region`` shares a row with this tensor."""
        return region.base_row < self.base_row + self.rows and self.base_row < region.end_row

    def slice(self, first_row: int, rows: int) -> Resident:
        """Rows ``first_row .. first_row + rows - 1`` of this tensor, as a
        resident of their own (same pack, format and step): an input of a
        step that reads one part, such as one third of a QKV output, or an
        ``at`` target a step writes.

        Raises:
            ConfigurationError: The rows are not this tensor's.
        """
        if first_row < 0 or rows < 1 or first_row + rows > self.rows:
            raise ConfigurationError(
                f"a slice of rows {first_row}..{first_row + rows - 1} of a resident tensor "
                f"of {self.rows} rows"
            )
        return Resident(
            self.base_row + first_row, rows, self.pack, self.format, self.step, parent=self
        )

    def root(self) -> Resident:
        """The whole tensor this is (a slice of)."""
        return self if self.parent is None else self.parent.root()


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
        self._live: list[Resident] = []
        self._held: list[Resident] = []
        self._fresh: ReferenceModel | None = None  # reset values, for reset=

    def hold(self, resident: Resident) -> Resident:
        """Keep ``resident`` in XMEM until :meth:`release`: later steps place
        their regions around it.

        Returns:
            ``resident``.

        Raises:
            ConfigurationError: It is stale, or another chain's.
        """
        if not self._is_live(resident):
            raise ConfigurationError(
                f"cannot hold rows {_rows(resident)}: the resident tensor is no longer in "
                "XMEM, or it is not this chain's"
            )
        if not _among(resident, self._live):
            self._live.append(resident)  # a slice outlives a write to its parent's other rows
        if not _among(resident, self._held):
            self._held.append(resident)
        return resident

    def reserve(
        self, rows: int, *, pack: str | None = None, fmt: ElementFormat | None = None
    ) -> Resident:
        """Hold ``rows`` free rows, placed by first fit around the live
        residents, for steps to write into (``at`` with its :meth:`Resident.slice`
        values), such as one tensor many one-row steps fill. Its contents are
        whatever XMEM held. Release it as any held resident.

        Args:
            rows: The rows.
            pack: The pack transform its writers lay it out with.
            fmt: Its stored elements' format; the chain's by default.

        Raises:
            ConfigurationError: ``rows`` is under 1, or they do not fit in XMEM.
        """
        fmt = self.format if fmt is None else fmt
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            raise ConfigurationError(f"reserve takes rows >= 1, not {rows!r}")
        base = 0
        for t in sorted(self._live, key=lambda t: t.base_row):
            if base + rows <= t.base_row:
                break
            base = max(base, t.base_row + t.rows)
        capacity = addressable_rows(self.format)  # rows are the chain format's
        if base + rows > capacity:
            raise ConfigurationError(
                f"reserve: {rows} rows do not fit in XMEM ({capacity} "
                f"{self.format.value} rows) around the live residents"
            )
        resident = Resident(base, rows, pack, fmt, self._steps)
        self._live.append(resident)
        self._held.append(resident)
        return resident

    def _is_live(self, resident: Resident) -> bool:
        """Whether ``resident`` (or the tensor it is a slice of) is in XMEM."""
        current: Resident | None = resident
        while current is not None:
            if _among(current, self._live):
                return True
            current = current.parent
        return False

    def release(self, resident: Resident) -> None:
        """Let later steps place regions over ``resident`` again.

        Raises:
            ConfigurationError: The chain does not hold it.
        """
        if not _among(resident, self._held):
            raise ConfigurationError(f"rows {_rows(resident)}: the chain does not hold them")
        self._held = [t for t in self._held if t is not resident]

    def _reset(self, banks: tuple[str, ...]) -> None:
        """Write every register of ``banks`` with its reset value: a fresh
        model's (CR0 and CR1 are hardwired)."""
        if not banks:
            return
        if self._fresh is None:
            self._fresh = ReferenceModel(self.format)
        for bank in banks:
            for index in range(REGISTER_DEFINITIONS[bank]["count"]):
                if bank != "cr" or index not in CR_READ_ONLY_INITIAL_VALUES:
                    self.model.write_register(bank, self._fresh.read_register(bank, index), index)

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
        at: Mapping[str, int | Resident] | None = None,
        reset: Iterable[str] = (),
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
            at: Base rows for regions this step writes (an array input, an
                output, a scratch region), by name, instead of first fit; or,
                for an output, a live ``Resident`` (a slice of a
                :meth:`reserve`'d tensor) to write into, laid out as the
                output is. That resident, and the tensor it is a slice of,
                stay live; the step's output is it.
            reset: Register banks (``r_mask``, ``lr``, ``r_acc``, ... as
                ``ReferenceModel.read_register`` names them) set to their reset
                values before the step: every register of each, CR0 and CR1
                aside. Done before the step's placement is checked, so a
                refused step may have reset them.

        Returns:
            The run, and its outputs as ``Resident`` tensors.

        Raises:
            ConfigurationError: An input is missing or names no input; an
                array input is not finite numbers of its shape; a
                ``Resident``'s pack, format or rows differ from what the kernel
                declares, or it moves an input ``kernel.toml`` pins to a row;
                or a ``Resident`` is stale (a later step placed a region over
                it) or another chain's; ``at`` names no tensor of the case (its
                ``when``), a resident input, or one ``kernel.toml`` pins elsewhere.
            PlanError: The case is wrong, a ``Resident`` overlaps another
                region (placement), or a region is pinned on, or would spill
                onto, a held resident.
        """
        manifest = default_registry().kernel(kernel) if isinstance(kernel, str) else kernel
        decl = CaseDecl(_STEP_CASE, MappingProxyType(dict(params)))
        plan = plan_case(manifest, params, self.format, case=decl)
        targets = self._targets(plan, at or {})
        placed = _checked_at(
            plan,
            {n: v.base_row if isinstance(v, Resident) else v for n, v in (at or {}).items()},
            inputs,
        )
        banks = _checked_banks(manifest, reset)
        pinned, arrays = self._split_inputs(plan, inputs)
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
        rendered = None if manifest.stages is not None else assemble(plan, self._cache).rendered
        spill = aaq_spill_rows(plan, rendered)
        reserved = _reserved(self._held, inputs, targets, spill)
        avoid = _avoid(reserved, targets, spill)
        written = _regions_written(plan, pinned, placed, reserved, avoid, spill)
        _check_reserved(plan, written, reserved)
        shared = SharedModel(
            self.model,
            MappingProxyType(arrays),
            MappingProxyType(pinned),
            check,
            MappingProxyType(placed),
            MappingProxyType(reserved),
            avoid,
        )
        self._reset(banks)
        options = self._step_options()
        self._steps += 1
        try:
            result = run_kernel(
                manifest, case, self.format, options, cache=self._cache, shared=shared
            )
        finally:
            # Even when the step raised: it may have written its regions first.
            self._forget_written(targets, written, spill)
        resident = {**_resident(result, plan, self._steps - 1), **targets}
        if result.placement is None or not result.halted:
            resident = {}
        self._live.extend(t for t in resident.values() if not _among(t, targets.values()))
        return ChainStep(result, MappingProxyType(resident))

    def _targets(self, plan: CasePlan, at: Mapping[str, int | Resident]) -> dict[str, Resident]:
        """The ``Resident`` values of ``at``: live, and laid out as the outputs."""
        targets = {n: v for n, v in at.items() if isinstance(v, Resident)}
        for name, target in targets.items():
            _check_layout(name, target, plan, role="output")
            if not self._is_live(target):
                raise ConfigurationError(
                    f"{plan.manifest.name}: at {name}'s resident tensor (rows {_rows(target)}) "
                    "is no longer in XMEM, or it is not this chain's"
                )
        return targets

    def _split_inputs(
        self, plan: CasePlan, inputs: Mapping[str, Array | Resident]
    ) -> tuple[dict[str, int], dict[str, Array]]:
        """The resident inputs' rows (checked live and laid out as declared),
        and the array inputs (checked)."""
        pinned: dict[str, int] = {}
        arrays: dict[str, Array] = {}
        for name, value in inputs.items():
            tensor = plan.manifest.tensors.get(name)
            if tensor is None or tensor.role != "input":
                # No input of the kernel: the runner names every wrong input.
                if isinstance(value, Resident):
                    pinned[name] = value.base_row
                else:
                    arrays[name] = value
            elif isinstance(value, Resident):
                _check_layout(name, value, plan)
                if not self._is_live(value):
                    raise ConfigurationError(
                        f"{plan.manifest.name}: input {name}'s resident tensor (rows "
                        f"{_rows(value)}) is no longer in XMEM: a later step wrote over it, "
                        "or it is not this chain's"
                    )
                pinned[name] = value.base_row
            else:
                arrays[name] = checked_input(plan, name, value)
        return pinned, arrays

    def _forget_written(
        self, targets: Mapping[str, Resident], written: list[Region], spill: int
    ) -> None:
        """Drop the live residents a step wrote over. A target written into,
        and the tensor it is part of, stay, unless the step wrote the rest of
        that tensor too; a target is then live in its own right."""
        into = [(t.base_row, t.base_row + t.rows + spill) for t in targets.values()]
        other = [r for r in written if not any(a <= r.base_row and r.end_row <= b for a, b in into)]
        kept = [a for t in targets.values() for a in _lineage(t)]
        self._live = [
            t
            for t in self._live
            if not any(t.overlaps(r) for r in (other if _among(t, kept) else written))
        ]
        self._live.extend(t for t in targets.values() if not self._is_live(t))


def _checked_at(
    plan: CasePlan, at: Mapping[str, int], inputs: Mapping[str, Array | Resident]
) -> dict[str, int]:
    """``at``, once each name is a region the step writes and each row is a row."""
    manifest = plan.manifest
    for name, row in at.items():
        if name not in manifest.tensors:
            raise ConfigurationError(
                f"{manifest.name}: at names {name!r}, which is no tensor of this case "
                f"(tensors: {', '.join(manifest.tensors)})"
            )
        if isinstance(inputs.get(name), Resident):
            raise ConfigurationError(
                f"{manifest.name}: at names input {name}, which is resident: it stays where it is"
            )
        if not isinstance(row, int) or isinstance(row, bool) or row < 0:
            raise ConfigurationError(f"{manifest.name}: at {name} = {row!r}, not a row")
        pin = plan.pins.get(name)
        if pin is not None and pin != row:
            raise ConfigurationError(
                f"{manifest.name}: {name} is pinned to row {pin} in kernel.toml; at gives {row}"
            )
    return dict(at)


def _checked_banks(manifest: KernelManifest, reset: Iterable[str]) -> tuple[str, ...]:
    """``reset``, once each is a register bank."""
    if isinstance(reset, str):
        raise ConfigurationError(f"{manifest.name}: reset takes bank names, not {reset!r}")
    banks = tuple(reset)
    for bank in banks:
        if bank not in REGISTER_DEFINITIONS:
            raise ConfigurationError(
                f"{manifest.name}: reset names {bank!r}, no register bank "
                f"(banks: {', '.join(sorted(REGISTER_DEFINITIONS))})"
            )
    return banks


def _reserved(
    held: list[Resident],
    inputs: Mapping[str, Array | Resident],
    targets: Mapping[str, Resident],
    spill: int,
) -> dict[str, tuple[int, int]]:
    """The rows of the held tensors a step may not place a region on, nor
    spill onto: all of each, but the parts the step reads (its resident
    inputs) and writes (its ``at`` targets, with the rows their ``ST.AAQ``
    spill reaches, rewritten when slices are written in row order)."""
    used = [(v, 0) for v in inputs.values() if isinstance(v, Resident)]
    used += [(t, spill) for t in targets.values()]
    out: dict[str, tuple[int, int]] = {}
    for tensor in held:
        parts = [(tensor.base_row, tensor.base_row + tensor.rows)]
        for v, extra in used:
            if not _related(tensor, v):
                continue
            lo, hi = v.base_row, v.base_row + v.rows + extra
            parts = [
                piece
                for a, b in parts
                for piece in ((a, min(b, lo)), (max(a, hi), b))
                if piece[0] < piece[1]
            ]
        for a, b in parts:
            out[f"held tensor (rows {a}..{b - 1})"] = (a, b - a)
    return out


def _avoid(
    reserved: Mapping[str, tuple[int, int]], targets: Mapping[str, Resident], spill: int
) -> tuple[tuple[int, int], ...]:
    """Rows first fit leaves for a store's spill: below each reserved range,
    and after each target (its own tensor's rows)."""
    avoid = [(max(0, base - spill), min(spill, base)) for base, _ in reserved.values()]
    avoid += [(t.base_row + t.rows, spill) for t in targets.values()]
    return tuple(a for a in avoid if a[1] > 0)


def _check_reserved(
    plan: CasePlan, written: list[Region], reserved: Mapping[str, tuple[int, int]]
) -> None:
    """Refuse a step whose writes reach a held tensor: a pinned region, or
    its spill (a scratch region's is no placement error)."""
    for region in written:
        for name, (base, rows) in reserved.items():
            if region.base_row < base + rows and base < region.end_row:
                tensor = plan.manifest.tensors[region.name]
                raise PlanError(
                    plan.manifest,
                    f"{tensor.section}.{tensor.name}.row",
                    f"{region.name} (rows {region.base_row}..{region.end_row - 1}, with "
                    f"its ST.AAQ spill) would write the {name}",
                )


def _lineage(resident: Resident) -> list[Resident]:
    """``resident``, the tensor it is a slice of, that one's, and so on."""
    out = []
    current: Resident | None = resident
    while current is not None:
        out.append(current)
        current = current.parent
    return out


def _related(a: Resident, b: Resident) -> bool:
    """Whether ``a`` and ``b`` are parts of one tensor and share a row."""
    return (
        a.root() is b.root()
        and a.base_row < b.base_row + b.rows
        and b.base_row < a.base_row + a.rows
    )


def _among(resident: Resident, residents: Iterable[Resident]) -> bool:
    """Whether ``resident`` is one of ``residents`` (the object, not an equal one)."""
    return any(resident is t for t in residents)


def _rows(resident: Resident) -> str:
    return f"{resident.base_row}..{resident.base_row + resident.rows - 1}"


def _regions_written(
    plan: CasePlan,
    pinned: Mapping[str, int],
    placed: Mapping[str, int],
    reserved: Mapping[str, tuple[int, int]],
    avoid: tuple[tuple[int, int], ...],
    spill: int,
) -> list[Region]:
    """The rows a step may write: its regions (placed around the ``reserved``
    rows) but its resident inputs, each element output and scratch region
    with the ``spill`` rows after it; none when placement refuses the step,
    which then fails before it writes."""
    tensors = plan.manifest.tensors
    try:
        placement = place(
            plan,
            {n: stored_rows(t, plan) for n, t in tensors.items()},
            pins={**pinned, **placed},
            reserved=reserved,
            avoid=avoid,
        )
    except (ValueError, PlanError):
        return []
    return [
        replace(r, rows=r.rows + spill)
        if r.role == "scratch" or (r.role == "output" and tensors[r.name].store == "element")
        else r
        for r in placement.regions.values()
        if r.name not in pinned
    ]


def _check_layout(name: str, value: Resident, plan: CasePlan, role: str = "input") -> None:
    manifest = plan.manifest
    tensor = manifest.tensors.get(name)
    if tensor is None or tensor.role != role:
        raise ConfigurationError(
            f"{manifest.name}: at {name} is a resident tensor, so {name} must be an output "
            "of this case, to write into it"
        )
    want = (tensor.pack, stored_format(tensor, plan.format), stored_rows(tensor, plan))
    have = (value.pack, value.format, value.rows)
    pin = plan.pins.get(name)
    if pin is not None and value.base_row != pin:
        raise ConfigurationError(
            f"{manifest.name}: {role} {name} is pinned to row {pin} in kernel.toml; "
            f"the resident tensor is at row {value.base_row}"
        )
    if have != want:
        raise ConfigurationError(
            f"{manifest.name}: {role} {name} is laid out as pack={want[0]!r}, "
            f"format={want[1].value}, {want[2]} rows; the resident tensor is "
            f"pack={have[0]!r}, format={have[1].value}, {have[2]} rows"
        )


def _resident(result: RunResult, plan: CasePlan, step: int) -> Mapping[str, Resident]:
    """Each output of ``result`` (of its case: ``when``), where its run left it."""
    if result.placement is None or not result.halted:
        return MappingProxyType({})
    out = {}
    for tensor in plan.manifest.tensors.values():
        if tensor.role == "output":
            region = result.placement.region(tensor.name)
            out[tensor.name] = Resident(
                region.base_row,
                region.rows,
                tensor.pack,
                stored_format(tensor, plan.format),
                step,
            )
    return MappingProxyType(out)
