"""Logical tensors to XMEM rows for the case's variant, and back (E06-05, Q83).

:func:`pack` turns a declared tensor's array into the bytes of its region
(and, in NX8, each row's metadata word): the exact shape is checked (L15: no
truncation, no padding of the input), a :class:`PackTransform` lays it out in
rows of 128 elements, and the codec rounds and encodes the rows for the case's
variant (NX8: each row quantized with its own scale, ADR 0007).
:func:`unpack` reverses it for an output, reading each row with its metadata.
:func:`held` is the two together: a tensor as the variant holds it.

A transform reshapes **one** tensor and sees nothing else: no other tensor, no
XMEM, no CR. That is Q83's rule made structural. :data:`DEFAULT` (row-major:
the last dimension across the lanes) serves most tensors; ``pack = "<name>"``
names another, found first in the kernel folder's ``pack.py``, then in the
family folder's ``pack.py`` (``kernels/<family>/pack.py``, shared by the
family's kernels; both loaded from their files), then in
``ipu_kernels.<family>.pack``, then in ``ipu_kernels.pack`` (imported by name,
as plugins: Q62; the harness has no import of ``ipu_kernels``).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

import numpy as np

from ipu_harness._codec import Array, decode_rows, encode_rows
from ipu_harness._plugins import import_named, load_file
from ipu_harness.manifest import TensorDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.geometry import LANES

__all__ = [
    "DEFAULT",
    "PackTransform",
    "Packed",
    "RowMajor",
    "held",
    "logical",
    "pack",
    "region_fill",
    "stored_rows",
    "transform_for",
    "unpack",
]

T = TypeVar("T")


class PackTransform(Protocol):
    """Lays ONE logical tensor out in rows of 128 elements, and back."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """The rows a tensor of ``shape`` takes in ``plan`` (a padded or tiled
        layout may depend on a parameter, not only the shape). A transform
        whose ``rows`` takes the shape only is still called with it; one
        whose second positional parameter means something else (``rows(shape,
        scale=1)``) would be given the plan there, so name it ``plan`` or
        make it keyword-only."""
        ...

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as a ``(rows, 128)`` array, zero-padded."""
        ...

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor that ``r`` (``(rows, 128)``) holds."""
        ...


class RowMajor:
    """The last dimension across the lanes, zero-padded to a whole number of
    rows (``ceil(n / 128)`` rows per index of the other dimensions)."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """``ceil(n / 128)`` rows for each index of the other dimensions."""
        return math.prod(shape[:-1]) * -(-shape[-1] // LANES)

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``, each last-dimension run zero-padded to whole rows."""
        width = -(-a.shape[-1] // LANES) * LANES
        padded = np.zeros((*a.shape[:-1], width), dtype=np.float64)
        padded[..., : a.shape[-1]] = a
        return padded.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`)."""
        width = -(-shape[-1] // LANES) * LANES
        return r.reshape(*shape[:-1], width)[..., : shape[-1]]


DEFAULT: PackTransform = RowMajor()

#: The file a kernel folder names its own transforms in.
PACK_FILE = "pack.py"


_PARTS = ("rows", "to_rows", "from_rows")


def _named(source: object, name: str, where: str) -> PackTransform | None:
    """``source``'s attribute ``name`` if it has one: a transform, or a ``ValueError``."""
    found = getattr(source, name, None)
    if found is None:
        return None
    if not all(callable(getattr(found, part, None)) for part in _PARTS):
        raise ValueError(f"{where} defines {name}, but not as a transform ({', '.join(_PARTS)})")
    return cast(PackTransform, found)


def _local(path: Path, name: str) -> PackTransform | None:
    """Transform ``name`` from a kernel or family folder's ``pack.py``, if it defines one."""
    return _named(load_file(path), name, str(path))


def _shared(module: str, name: str) -> PackTransform | None:
    """Transform ``name`` from ``module``, if the module exists and defines one."""
    found = import_named(module)
    return None if found is None else _named(found, name, module)


def transform_for(
    tensor: TensorDecl, plan: CasePlan, *, package: str = "ipu_kernels"
) -> PackTransform:
    """The transform of ``tensor``: :data:`DEFAULT`, or the one its ``pack`` names.

    Args:
        tensor: The declared tensor.
        plan: Its case's plan (for the kernel's folder and family).
        package: Where family and shared transforms live (a test names another).

    Returns:
        The transform.

    Raises:
        PlanError: Neither the kernel's ``pack.py``, its family folder's, its
            family's package nor the shared library defines it.
    """
    if tensor.pack is None:
        return DEFAULT
    family = plan.manifest.directory.parent.name
    key = f"{tensor.section}.{tensor.name}.pack"
    try:
        found = None
        for local in (
            plan.manifest.directory / PACK_FILE,
            plan.manifest.directory.parent / PACK_FILE,
        ):
            if found is None and local.is_file():
                found = _local(local, tensor.pack)
        for module in (f"{package}.{family}.pack", f"{package}.pack"):
            if found is None:
                found = _shared(module, tensor.pack)
    except Exception as error:
        raise PlanError(
            plan.manifest, key, f"looking up {tensor.pack!r}: {type(error).__name__}: {error}"
        ) from error
    if found is None:
        raise PlanError(
            plan.manifest,
            key,
            f"no transform {tensor.pack!r} in the kernel's {PACK_FILE}, the family's "
            f"{family}/{PACK_FILE}, {package}.{family}.pack or {package}.pack",
        )
    return found


# --- pack and unpack ----------------------------------------------------------------------


@dataclass(frozen=True)
class Packed:
    """A region's contents: its rows' bytes and, in NX8, each row's metadata.

    Attributes:
        data: ``rows x row_bytes`` bytes.
        metadata: Each row's metadata word (NX8), or ``None`` (FP32, or bytes
            a scratch function gave: their rows' metadata is left 0).
    """

    data: bytes
    metadata: tuple[int, ...] | None = None


def _laid_out(
    tensor: TensorDecl, plan: CasePlan, package: str, step: Callable[[PackTransform], T]
) -> T:
    """``step(transform)``; a shape the transform cannot lay out is located at ``pack``."""
    transform = transform_for(tensor, plan, package=package)
    try:
        return step(transform)
    except ValueError as error:
        raise PlanError(plan.manifest, f"{tensor.section}.{tensor.name}.pack", str(error)) from None


def _rows(transform: PackTransform, shape: tuple[int, ...], plan: CasePlan) -> int:
    """``transform.rows(shape, plan)``; ``rows(shape)`` for a transform written
    before ``rows`` took the plan.

    A parameter named ``plan`` (keyword-only, or set by ``functools.partial``;
    not a positional-only one) gets it by name; two positional parameters, or ``*args``, get it second;
    otherwise only the shape is passed. A callable with no signature (a
    builtin) is given both.
    """
    try:
        params = inspect.signature(transform.rows).parameters
    except (TypeError, ValueError):
        return int(transform.rows(shape, plan))
    named = params.get("plan")
    if named is not None and named.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return int(transform.rows(shape, plan=plan))
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values())
    if len(positional) >= 2 or variadic:
        return int(transform.rows(shape, plan))
    return int(transform.rows(shape))  # type: ignore[call-arg]


def stored_rows(tensor: TensorDecl, plan: CasePlan, *, package: str = "ipu_kernels") -> int:
    """The XMEM rows ``tensor`` takes (E06-04's row counts).

    Args:
        tensor: The declared tensor.
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        Its transform's rows.

    Raises:
        PlanError: The transform is unknown or cannot lay out the planned shape.
    """
    shape = plan.shapes[tensor.name]
    return _laid_out(tensor, plan, package, lambda t: _rows(t, shape, plan))


def pack(tensor: TensorDecl, a: Array, plan: CasePlan, *, package: str = "ipu_kernels") -> Packed:
    """The contents of ``tensor``'s region holding ``a``, for ``plan.format``.

    Args:
        tensor: The declared tensor.
        a: Its logical values; exactly the planned shape (L15).
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        ``stored_rows`` rows: the transform's rows, rounded to the variant and
        encoded (NX8: with each row's metadata word).

    Raises:
        PlanError: ``a``'s shape is not the planned one (oversize included), or
            the transform is unknown, cannot lay it out, or lays out another
            number of rows than its ``rows`` gives (located at ``pack``).
    """
    shape = plan.shapes[tensor.name]
    if a.shape != shape:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.shape",
            f"{tensor.name} has shape {a.shape}; the case's is {shape}",
        )
    values = np.asarray(a, dtype=np.float64)
    rows = _laid_out(tensor, plan, package, lambda t: _to_rows(t, values, plan))
    return Packed(*encode_rows(rows.reshape(-1, LANES), plan.format))


def logical(
    tensor: TensorDecl, rows: Array, plan: CasePlan, *, package: str = "ipu_kernels"
) -> Array:
    """The planned-shape tensor that ``rows`` (``(stored_rows, 128)``, any
    per-element values) lay out, by ``tensor``'s transform."""
    shape = plan.shapes[tensor.name]
    grid = np.asarray(rows, dtype=np.float64).reshape(-1, LANES)
    return _laid_out(tensor, plan, package, lambda t: t.from_rows(grid, plan, shape))


def held(tensor: TensorDecl, a: Array, plan: CasePlan, *, package: str = "ipu_kernels") -> Array:
    """``a`` as ``tensor``'s region holds it in ``plan.format``: packed and
    unpacked (NX8: each row quantized with its own scale, so a value depends on
    its row's other values; FP32: rounded to float32).

    Raises:
        PlanError: As :func:`pack`.
    """
    packed = pack(tensor, a, plan, package=package)
    return unpack(tensor, packed.data, plan, packed.metadata, package=package)


def _to_rows(transform: PackTransform, a: Array, plan: CasePlan) -> Array:
    """``transform.to_rows(a, plan)``, which must fill exactly the rows its
    ``rows`` gives: placement sized the region by those, so a transform that
    lays out more (one thinking in an element size other than the format's)
    would write over the next region.

    Raises:
        ValueError: It laid out another number of elements.
    """
    want = _rows(transform, a.shape, plan)
    rows = np.asarray(transform.to_rows(a, plan))
    if rows.size != want * LANES:
        raise ValueError(
            f"the transform laid out {rows.size} elements ({rows.size / LANES:g} rows of "
            f"{LANES}), but its rows() gives {want} rows ({want * LANES} elements)"
        )
    return rows


def unpack(
    tensor: TensorDecl,
    raw: bytes,
    plan: CasePlan,
    metadata: tuple[int, ...] | None = None,
    *,
    package: str = "ipu_kernels",
) -> Array:
    """The logical values ``tensor``'s region holds (inverse of :func:`pack`).

    Args:
        tensor: The declared tensor.
        raw: The region's bytes, exactly ``stored_rows x row_bytes``.
        plan: The checked case.
        metadata: Each row's metadata word, read with the bytes (NX8; ``None``
            reads every row's as 0).
        package: Where family and shared transforms live.

    Returns:
        A float64 array of the planned shape.

    Raises:
        PlanError: ``raw`` (or ``metadata``) is not the region's size, or the
            transform is unknown or cannot lay the shape out.
    """
    shape = plan.shapes[tensor.name]
    rows = stored_rows(tensor, plan, package=package)
    want = rows * plan.format.row_bytes
    if len(raw) != want or (metadata is not None and len(metadata) != rows):
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.size",
            f"{tensor.name}'s region holds {want} bytes ({rows} rows) in "
            f"{plan.format.value}, given {len(raw)}",
        )
    values = decode_rows(raw, plan.format, metadata)
    return _laid_out(tensor, plan, package, lambda t: t.from_rows(values, plan, shape))


def region_fill(tensor: TensorDecl, plan: CasePlan, rows: int) -> Packed:
    """A scratch region's starting contents.

    With ``bytes = "file.py:fn"``, what ``fn(plan)`` returns, unchanged (a lane
    mask no element value can carry), its rows' metadata left 0. Otherwise its
    ``fill`` in every element of every row it occupies, padding lanes included
    (not through the pack transform, which pads with zeros), held by the
    variant (NX8: with each row's metadata).

    Args:
        tensor: The scratch tensor.
        plan: The checked case.
        rows: The rows its region occupies (``Placement``).

    Returns:
        ``rows x row_bytes`` bytes, and the rows' metadata.

    Raises:
        PlanError: The function's result is not bytes of exactly that size.
    """
    region_bytes = rows * plan.format.row_bytes
    if tensor.content is None:
        fill = plan.fills.get(tensor.name, tensor.fill)  # an expression's value, from the plan
        return Packed(*encode_rows(np.full((rows, LANES), fill, dtype=np.float64), plan.format))
    file, _, name = tensor.content.partition(":")
    data = getattr(load_file(plan.manifest.directory / file), name)(plan)
    key = f"{tensor.section}.{tensor.name}.bytes"
    if not isinstance(data, (bytes, bytearray)):
        raise PlanError(
            plan.manifest, key, f"{tensor.content} returned {type(data).__name__}, not bytes"
        )
    if len(data) != region_bytes:
        raise PlanError(
            plan.manifest,
            key,
            f"{tensor.content} returned {len(data)} bytes; the region holds {region_bytes} "
            f"({rows} rows of {plan.format.value})",
        )
    return Packed(bytes(data))
