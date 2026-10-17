"""Logical tensors to XMEM bytes for the active format, and back (E06-05, Q83).

:func:`pack` turns a declared tensor's array into the bytes of its region:
the exact shape is checked (L15: no truncation, no padding of the input), a
:class:`PackTransform` lays it out in rows of 128 elements, and the codec
rounds and encodes the elements for ``plan.format``. :func:`unpack` reverses
it for an output: an ``element`` output decodes in the format, an
``accumulator`` output in the format's accumulator format (FP32 for FP32 and
FP8, INT32 for INT8), whose vectors span several narrow rows (E06-04).

A transform reshapes **one** tensor and sees nothing else: no other tensor, no
XMEM, no CR. That is Q83's rule made structural. :data:`DEFAULT` (row-major:
the last dimension across the lanes) serves most tensors; ``pack = "<name>"``
names another, found first in the kernel folder's ``pack.py`` (loaded from
its file), then in ``ipu_kernels.<family>.pack``, then in ``ipu_kernels.pack``
(imported by name, as plugins: Q62; the harness has no import of
``ipu_kernels``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar, cast

import numpy as np

from ipu_harness._codec import Array, decode, encode
from ipu_harness._plugins import import_named, load_file
from ipu_harness.manifest import TensorDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES

__all__ = [
    "DEFAULT",
    "PackTransform",
    "RowMajor",
    "pack",
    "region_fill",
    "stored_format",
    "stored_rows",
    "transform_for",
    "unpack",
]

T = TypeVar("T")


class PackTransform(Protocol):
    """Lays ONE logical tensor out in rows of 128 elements, and back."""

    def rows(self, shape: tuple[int, ...]) -> int:
        """The rows a tensor of ``shape`` takes."""
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

    def rows(self, shape: tuple[int, ...]) -> int:
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
    """Transform ``name`` from a kernel folder's ``pack.py``, if it defines one."""
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
        PlanError: Neither the kernel's ``pack.py``, its family's nor the
            shared library defines it.
    """
    if tensor.pack is None:
        return DEFAULT
    family = plan.manifest.directory.parent.name
    local = plan.manifest.directory / PACK_FILE
    key = f"{tensor.section}.{tensor.name}.pack"
    try:
        found = _local(local, tensor.pack) if local.is_file() else None
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
            f"no transform {tensor.pack!r} in the kernel's {PACK_FILE}, "
            f"{package}.{family}.pack or {package}.pack",
        )
    return found


# --- pack and unpack ----------------------------------------------------------------------


def stored_format(tensor: TensorDecl, fmt: ElementFormat) -> ElementFormat:
    """The format a tensor's bytes are in.

    Args:
        tensor: The declared tensor.
        fmt: The case's element format.

    Returns:
        Its accumulator format (FP32 or INT32) for an ``accumulator`` output,
        else ``fmt``.
    """
    if tensor.store == "accumulator":
        return ElementFormat(FORMATS[fmt].accumulator.value)
    return fmt


def _laid_out(
    tensor: TensorDecl, plan: CasePlan, package: str, step: Callable[[PackTransform], T]
) -> T:
    """``step(transform)``; a shape the transform cannot lay out is located at ``pack``."""
    transform = transform_for(tensor, plan, package=package)
    try:
        return step(transform)
    except ValueError as error:
        raise PlanError(plan.manifest, f"{tensor.section}.{tensor.name}.pack", str(error)) from None


def stored_rows(tensor: TensorDecl, plan: CasePlan, *, package: str = "ipu_kernels") -> int:
    """The XMEM rows ``tensor`` takes as stored (E06-04's row counts).

    Args:
        tensor: The declared tensor.
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        Its transform's rows, times the accumulator stride for an
        ``accumulator`` output.

    Raises:
        PlanError: The transform is unknown or cannot lay out the planned shape.
    """
    shape = plan.shapes[tensor.name]
    rows = _laid_out(tensor, plan, package, lambda t: t.rows(shape))
    stored = FORMATS[stored_format(tensor, plan.format)].element_bytes
    return rows * stored // FORMATS[plan.format].element_bytes


def pack(tensor: TensorDecl, a: Array, plan: CasePlan, *, package: str = "ipu_kernels") -> bytes:
    """The bytes of ``tensor``'s region holding ``a``, for ``plan.format``.

    Args:
        tensor: The declared tensor.
        a: Its logical values; exactly the planned shape (L15).
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        ``stored_rows x row_bytes`` bytes: the transform's rows, each value
        rounded to the stored format and encoded.

    Raises:
        PlanError: ``a``'s shape is not the planned one (oversize included), or
            the transform is unknown or cannot lay it out.
    """
    shape = plan.shapes[tensor.name]
    if a.shape != shape:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.shape",
            f"{tensor.name} has shape {a.shape}; the case's is {shape}",
        )
    values = np.asarray(a, dtype=np.float64)
    rows = _laid_out(tensor, plan, package, lambda t: t.to_rows(values, plan))
    return encode(rows, stored_format(tensor, plan.format))


def unpack(
    tensor: TensorDecl, raw: bytes, plan: CasePlan, *, package: str = "ipu_kernels"
) -> Array:
    """The logical values ``tensor``'s region holds (inverse of :func:`pack`).

    An ``accumulator`` output is read in the accumulator format: one vector of
    128 accumulator elements spans ``accumulator_bytes x 128 / row_bytes``
    narrow rows, the stride placement allocated (E06-04, :func:`stored_rows`).

    Args:
        tensor: The declared tensor.
        raw: The region's bytes, exactly ``stored_rows x row_bytes``.
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        A float64 array of the planned shape.

    Raises:
        PlanError: ``raw`` is not the region's size, or the transform is
            unknown or cannot lay the shape out.
    """
    shape = plan.shapes[tensor.name]
    want = stored_rows(tensor, plan, package=package) * FORMATS[plan.format].row_bytes
    stored = stored_format(tensor, plan.format)
    if len(raw) != want:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.size",
            f"{tensor.name}'s region holds {want} bytes in {stored.value}, given {len(raw)}",
        )
    rows = decode(raw, stored).reshape(-1, LANES)
    return _laid_out(tensor, plan, package, lambda t: t.from_rows(rows, plan, shape))


def region_fill(tensor: TensorDecl, plan: CasePlan, rows: int) -> bytes:
    """A scratch region's starting bytes.

    With ``bytes = "file.py:fn"``, what ``fn(plan)`` returns, unchanged (a lane
    mask no element value can carry). Otherwise its ``fill`` in every element of
    every row it occupies, padding lanes included (not through the pack
    transform, which pads with zeros), in its stored format.

    Args:
        tensor: The scratch tensor.
        plan: The checked case.
        rows: The rows its region occupies (``Placement``).

    Returns:
        ``rows x row_bytes`` bytes.

    Raises:
        PlanError: The function's result is not bytes of exactly that size.
    """
    region_bytes = rows * FORMATS[plan.format].row_bytes
    if tensor.content is None:
        fmt = stored_format(tensor, plan.format)
        elements = region_bytes // FORMATS[fmt].element_bytes
        return encode(np.full(elements, tensor.fill, dtype=np.float64), fmt)
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
    return bytes(data)
