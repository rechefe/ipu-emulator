"""Automatic XMEM region placement, in rows, with optional pins (E06-04, Q81).

:func:`place` gives every tensor of a :class:`~ipu_harness.params.CasePlan` a
base row: pinned regions (``row``, an int or an expression the plan
evaluated) first, then the rest in placement order, each at the lowest free
row it fits. Placement order is declaration order (inputs, outputs, scratch),
except that a tensor with ``after = "<t>"`` comes right after ``t``. The same
plan always gives the same bases, so a migrated kernel reproduces its old
layout (E07 proves it against the E02 digests). Bases reach the program only
through ``[abi]`` (E06-06); no kernel computes one. Placement only reserves
rows: the runner writes a scratch region's ``fill`` or ``bytes`` (E06-11).

Rows, not bytes (Q50): an element tensor needs the same rows in every format,
because a row is 128 elements wide whatever their size; only the capacity in
rows changes, as XMEM bytes are fixed (``ipu_isa.formats.addressable_rows``).
An output stored from the accumulator (``store = "accumulator"``) writes full
accumulator elements, so one vector takes ``accumulator_bytes x 128 /
row_bytes`` rows (4 in INT8, 1 in FP32).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ipu_as.session import AssemblySession
from ipu_harness.manifest import Role, TensorDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, addressable_rows
from ipu_isa.geometry import LANES
from ipu_isa.registers import REGISTER_DEFINITIONS


@dataclass(frozen=True)
class Region:
    """Where one tensor lives in XMEM.

    Attributes:
        name: The tensor's name.
        role: ``input``, ``output`` or ``scratch``.
        base_row: Its first row.
        rows: The rows it occupies.
    """

    name: str
    role: Role
    base_row: int
    rows: int

    @property
    def end_row(self) -> int:
        """One past its last row."""
        return self.base_row + self.rows


@dataclass(frozen=True)
class Placement:
    """Every region of a case, and the rows they span.

    Attributes:
        regions: The regions, by tensor name, in declaration order.
        rows_used: One past the last row any region occupies.
        capacity: The XMEM rows addressable in the case's format.
        reserved: The rows ``place`` kept free for others (``reserved``).
    """

    regions: Mapping[str, Region]
    rows_used: int
    capacity: int
    reserved: tuple[Region, ...] = ()

    def region(self, name: str) -> Region:
        """The region of tensor ``name``.

        Raises:
            KeyError: ``name`` is no tensor of the case.
        """
        return self.regions[name]


def default_rows(plan: CasePlan, tensor: TensorDecl) -> int:
    """The rows a tensor occupies when packing (E06-05) does not say.

    Args:
        plan: The checked case.
        tensor: One of its tensors.

    Returns:
        One row per 128 elements of the last dimension, for each index of the
        others; times ``accumulator_bytes x 128 / row_bytes`` for an output
        stored from the accumulator.
    """
    shape = plan.shapes[tensor.name]
    vectors = math.prod(shape[:-1]) * -(-shape[-1] // LANES)
    if tensor.store == "accumulator":
        info = FORMATS[plan.format]
        return vectors * (info.accumulator_bytes * LANES // info.row_bytes)
    return vectors


def placement_order(tensors: Iterable[TensorDecl]) -> list[TensorDecl]:
    """Declaration order, with each ``after`` tensor moved right behind its
    target (the loader checked the targets exist and form no cycle); one whose
    target this case lacks (``when``) keeps its declaration place.

    Args:
        tensors: The kernel's tensors, in declaration order.

    Returns:
        The order free regions are placed in.
    """
    declared = list(tensors)
    names = {t.name for t in declared}
    followers: dict[str, list[TensorDecl]] = {}
    for tensor in declared:
        if tensor.after in names:
            followers.setdefault(tensor.after, []).append(tensor)
    order: list[TensorDecl] = []

    def visit(tensor: TensorDecl) -> None:
        order.append(tensor)
        for follower in followers.get(tensor.name, ()):
            visit(follower)

    for tensor in declared:
        # A tensor whose target is absent from this case (``when``) keeps its
        # declaration place.
        if tensor.after not in names:
            visit(tensor)
    return order


def _key(tensor: TensorDecl) -> str:
    return f"{tensor.section}.{tensor.name}.row"


def _check_fits(plan: CasePlan, tensor: TensorDecl, region: Region, capacity: int) -> None:
    if region.end_row <= capacity:
        return
    where = (
        f"pinned at row {region.base_row} needs {region.rows} rows"
        if tensor.name in plan.pins
        else f"needs rows {region.base_row}..{region.end_row - 1}"
    )
    raise PlanError(
        plan.manifest,
        _key(tensor),
        f"{region.name} {where}, past the {capacity} rows of XMEM in {plan.format.value}",
    )


def _pinned(
    plan: CasePlan, rows: Mapping[str, int], capacity: int, extra: Mapping[str, int]
) -> list[Region]:
    pins: list[Region] = []
    for tensor in plan.manifest.tensors.values():
        row = extra.get(tensor.name, plan.pins.get(tensor.name))
        if row is None:
            continue
        region = Region(tensor.name, tensor.role, row, rows[tensor.name])
        _check_fits(plan, tensor, region, capacity)
        for other in pins:
            if region.base_row < other.end_row and other.base_row < region.end_row:
                raise PlanError(
                    plan.manifest,
                    _key(tensor),
                    f"{region.name} (rows {region.base_row}..{region.end_row - 1}) overlaps "
                    f"{other.name} (rows {other.base_row}..{other.end_row - 1})",
                )
        pins.append(region)
    return pins


def _first_fit(taken: list[Region], rows: int) -> int:
    """The lowest row where ``rows`` free rows start, around the regions ``taken``."""
    base = 0
    for region in sorted(taken, key=lambda r: r.base_row):
        if base + rows <= region.base_row:
            return base
        base = max(base, region.end_row)
    return base


def _row_counts(plan: CasePlan, row_counts: Mapping[str, int] | None) -> dict[str, int]:
    counts = dict(row_counts or {})
    for name, count in counts.items():
        if name not in plan.manifest.tensors:
            raise ValueError(f"row count for {name!r}, which is no tensor")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(
                f"row count {count!r} for {name!r}; a region takes a whole number of rows, at least 1"
            )
    return {
        name: counts.get(name, default_rows(plan, tensor))
        for name, tensor in plan.manifest.tensors.items()
    }


def place(
    plan: CasePlan,
    row_counts: Mapping[str, int] | None = None,
    *,
    pins: Mapping[str, int] | None = None,
    reserved: Mapping[str, tuple[int, int]] | None = None,
    avoid: Iterable[tuple[int, int]] = (),
) -> Placement:
    """Place every tensor of ``plan``. Takes the plan as checked; checks no parameter.

    Args:
        plan: The checked case.
        row_counts: Rows each tensor occupies as stored, from packing (E06-05),
            an accumulator output's stride included; a tensor left out takes
            :func:`default_rows`.
        pins: Base rows a caller fixes besides the ``row = N`` pins, by tensor
            name: a chained input already in XMEM (E06-15). Checked like them.
        reserved: Rows no region may take, as ``(base row, rows)`` by a name
            for messages: a tensor a chain holds in XMEM (E06-15). First fit
            places around them; a pin on them is an error; so is an
            ``ST.AAQ`` spill onto them (:func:`check_aaq_spill`).
        avoid: Rows first fit leaves free besides, as ``(base row, rows)``;
            a pin on them is fine: room for a store's spill
            (:func:`aaq_spill_rows`) below a reserved range, say.

    Returns:
        The placement, deterministic for the plan.

    Raises:
        PlanError: Pins overlap each other or a reserved range or run past
            capacity, or the regions do not fit.
        ValueError: ``row_counts`` or ``pins`` names no tensor, or
            ``row_counts`` gives one fewer than 1 row.
    """
    rows = _row_counts(plan, row_counts)
    capacity = addressable_rows(plan.format)
    extra = dict(pins or {})
    unknown = set(extra) - set(plan.manifest.tensors)
    if unknown:
        raise ValueError(f"pins for {sorted(unknown)}, which are no tensors")
    taken = _pinned(plan, rows, capacity, extra)
    blocked = [Region(name, "scratch", base, n) for name, (base, n) in (reserved or {}).items()]
    fenced = blocked + [Region("(avoided)", "scratch", base, n) for base, n in avoid]
    for region in taken:
        for other in blocked:
            if region.base_row < other.end_row and other.base_row < region.end_row:
                raise PlanError(
                    plan.manifest,
                    _key(plan.manifest.tensors[region.name]),
                    f"{region.name} (rows {region.base_row}..{region.end_row - 1}) overlaps "
                    f"{other.name} (rows {other.base_row}..{other.end_row - 1})",
                )
    for tensor in placement_order(plan.manifest.tensors.values()):
        if tensor.name not in plan.pins and tensor.name not in extra:
            need = rows[tensor.name]
            region = Region(tensor.name, tensor.role, _first_fit(taken + fenced, need), need)
            _check_fits(plan, tensor, region, capacity)
            taken.append(region)
    by_name = {r.name: r for r in taken}
    return Placement(
        regions=MappingProxyType({name: by_name[name] for name in plan.manifest.tensors}),
        rows_used=max((r.end_row for r in taken), default=0),
        capacity=capacity,
        reserved=tuple(blocked),
    )


#: ``ST.AAQ`` stores the whole of ``POST_AAQ_REG`` (OQ-35), more than one row
#: in an 8-bit format.
_AAQ_STORE_BYTES: int = REGISTER_DEFINITIONS["post_aaq_reg"]["size_bytes"]


def _stores_with_aaq(rendered: str) -> bool:
    """Whether the rendered program has an ``ST.AAQ`` instruction (comments
    and labels aside)."""
    return any(
        instruction["opcode"].token.value.upper() == "ST.AAQ"
        for word in AssemblySession("").parse_rendered(rendered)
        for instruction in word["instructions"]
    )


def aaq_spill_rows(plan: CasePlan, rendered: str | None) -> int:
    """The rows past an element output's last row that its stores write.

    ``ST.AAQ`` writes all of ``POST_AAQ_REG`` (OQ-35): more than one row in an
    8-bit format.

    Args:
        plan: The checked case.
        rendered: Its rendered program; ``None`` when there is no one program
            (a multi-stage kernel, E06-14), which is taken to store with
            ``ST.AAQ``.

    Returns:
        0 when the program has no ``ST.AAQ`` or the format's row holds the
        whole register; else the rows after the last one it writes.
    """
    spill = _AAQ_STORE_BYTES // FORMATS[plan.format].row_bytes - 1
    if spill <= 0 or (rendered is not None and not _stores_with_aaq(rendered)):
        return 0
    return spill


def check_aaq_spill(plan: CasePlan, placement: Placement, program: str) -> None:
    """Refuse a placement where ``ST.AAQ``'s spill would land on a region.

    ``ST.AAQ`` writes all of ``POST_AAQ_REG``, more than one row in an 8-bit
    format, so a store to an output's last row also writes (zeros to) the
    rows after it (OQ-35). When ``program`` (rendered) stores with
    ``ST.AAQ``, the rows after every element output must be free (of
    regions and of reserved rows) and inside XMEM. Accumulator outputs
    (``ST.ACC``) are not checked: their stride already holds a store. Nor is
    a scratch region a program stores to with ``ST.AAQ`` (a follow-up in
    OQ-35).

    Raises:
        PlanError: A region is in an output's spill, or the spill runs past
            XMEM.
    """
    spill = aaq_spill_rows(plan, program)
    if not spill:
        return
    capacity = addressable_rows(plan.format)
    for tensor in plan.manifest.tensors.values():
        if tensor.role != "output" or tensor.store != "element":
            continue
        end = placement.region(tensor.name).end_row
        hit = [
            r.name
            for r in (*placement.regions.values(), *placement.reserved)
            if r.name != tensor.name and r.base_row < end + spill and end < r.end_row
        ]
        if hit or end + spill > capacity:
            where = (
                f"which hold region {', '.join(hit)}"
                if hit
                else f"past XMEM's last row ({capacity - 1})"
            )
            raise PlanError(
                plan.manifest,
                _key(tensor),
                f"ST.AAQ writes {_AAQ_STORE_BYTES} bytes ({spill + 1} {plan.format.value} rows), "
                f"so a store to {tensor.name}'s last row writes rows {end}..{end + spill - 1}, "
                f"{where} (OQ-35)",
            )
