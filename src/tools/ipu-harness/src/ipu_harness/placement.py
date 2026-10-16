"""Automatic XMEM region placement, in rows, with optional pins (E06-04, Q81).

:func:`place` gives every tensor of a :class:`~ipu_harness.params.CasePlan` a
base row: pinned regions (``row = N``) first, then the rest in declaration
order (inputs, outputs, scratch), each at the lowest free row it fits. The same
plan always gives the same bases, so a migrated kernel reproduces its old
layout (E07 proves it against the E02 digests). Bases reach the program only
through ``[abi]`` (E06-06); no kernel computes one. Placement only reserves
rows: the runner writes a scratch region's ``fill`` (E06-11).

Rows, not bytes (Q50): an element tensor needs the same rows in every format,
because a row is 128 elements wide whatever their size; only the capacity in
rows changes, as XMEM bytes are fixed (``ipu_isa.formats.addressable_rows``).
An output stored from the accumulator (``store = "accumulator"``) writes full
accumulator elements, so one vector takes ``accumulator_bytes x 128 /
row_bytes`` rows (4 in INT8, 1 in FP32).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ipu_harness.manifest import Role, TensorDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, addressable_rows
from ipu_isa.geometry import LANES


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
    """

    regions: Mapping[str, Region]
    rows_used: int
    capacity: int

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


def _key(tensor: TensorDecl) -> str:
    return f"{tensor.section}.{tensor.name}.row"


def _check_fits(plan: CasePlan, tensor: TensorDecl, region: Region, capacity: int) -> None:
    if region.end_row <= capacity:
        return
    where = (
        f"pinned at row {region.base_row} needs {region.rows} rows"
        if tensor.row is not None
        else f"needs rows {region.base_row}..{region.end_row - 1}"
    )
    raise PlanError(
        plan.manifest,
        _key(tensor),
        f"{region.name} {where}, past the {capacity} rows of XMEM in {plan.format.value}",
    )


def _pinned(plan: CasePlan, rows: Mapping[str, int], capacity: int) -> list[Region]:
    pins: list[Region] = []
    for tensor in plan.manifest.tensors.values():
        if tensor.row is None:
            continue
        region = Region(tensor.name, tensor.role, tensor.row, rows[tensor.name])
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


def place(plan: CasePlan, row_counts: Mapping[str, int] | None = None) -> Placement:
    """Place every tensor of ``plan``. Takes the plan as checked; checks no parameter.

    Args:
        plan: The checked case.
        row_counts: Rows each tensor occupies as stored, from packing (E06-05),
            an accumulator output's stride included; a tensor left out takes
            :func:`default_rows`.

    Returns:
        The placement, deterministic for the plan.

    Raises:
        PlanError: Pins overlap or run past capacity, or the regions do not fit.
        ValueError: ``row_counts`` names no tensor, or gives one fewer than 1 row.
    """
    rows = _row_counts(plan, row_counts)
    capacity = addressable_rows(plan.format)
    taken = _pinned(plan, rows, capacity)
    for tensor in plan.manifest.tensors.values():
        if tensor.row is None:
            need = rows[tensor.name]
            region = Region(tensor.name, tensor.role, _first_fit(taken, need), need)
            _check_fits(plan, tensor, region, capacity)
            taken.append(region)
    by_name = {r.name: r for r in taken}
    return Placement(
        regions=MappingProxyType({name: by_name[name] for name in plan.manifest.tensors}),
        rows_used=max((r.end_row for r in taken), default=0),
        capacity=capacity,
    )
