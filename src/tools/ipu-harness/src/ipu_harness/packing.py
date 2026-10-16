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

import importlib.util
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from ipu_emu.errors import ConfigurationError
from ipu_harness._codec import Array, decode, encode
from ipu_harness.manifest import TensorDecl
from ipu_harness.params import CasePlan, PlanError
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES

__all__ = ["DEFAULT", "PackTransform", "RowMajor", "pack", "transform_for", "unpack"]


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
        return math.prod(shape[:-1]) * -(-shape[-1] // LANES)

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        width = -(-a.shape[-1] // LANES) * LANES
        padded = np.zeros((*a.shape[:-1], width), dtype=np.float64)
        padded[..., : a.shape[-1]] = a
        return padded.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        width = -(-shape[-1] // LANES) * LANES
        return r.reshape(*shape[:-1], width)[..., : shape[-1]]


DEFAULT: PackTransform = RowMajor()

#: The file a kernel folder names its own transforms in.
PACK_FILE = "pack.py"


def _named(source: object, name: str) -> PackTransform | None:
    found = getattr(source, name, None)
    return found if found is not None and callable(getattr(found, "to_rows", None)) else None


def _local(path: Path, name: str) -> PackTransform | None:
    """Transform ``name`` from a kernel folder's ``pack.py``, if it defines one."""
    spec = importlib.util.spec_from_file_location(f"_ipu_pack_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"{path}: cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _named(module, name)


def _shared(module: str, name: str) -> PackTransform | None:
    """Transform ``name`` from ``module``, if the module exists and defines one."""
    try:
        found = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name is not None and (module + ".").startswith(error.name + "."):
            return None
        raise
    return _named(found, name)


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
    found = _local(local, tensor.pack) if local.is_file() else None
    for module in (f"{package}.{family}.pack", f"{package}.pack"):
        found = found or _shared(module, tensor.pack)
    if found is None:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.pack",
            f"no transform {tensor.pack!r} in the kernel's {PACK_FILE}, "
            f"{package}.{family}.pack or {package}.pack",
        )
    return found


# --- pack and unpack ----------------------------------------------------------------------


def _stored_format(tensor: TensorDecl, fmt: ElementFormat) -> ElementFormat:
    """The format an output's bytes are in: the element format, or its accumulator's."""
    if tensor.store == "accumulator":
        return ElementFormat(FORMATS[fmt].accumulator.value)
    return fmt


def pack(tensor: TensorDecl, a: Array, plan: CasePlan, *, package: str = "ipu_kernels") -> bytes:
    """The bytes of ``tensor``'s region holding ``a``, for ``plan.format``.

    Args:
        tensor: The declared tensor.
        a: Its logical values; exactly the planned shape (L15).
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        ``rows * row_bytes`` bytes: the transform's rows, each value rounded to
        the format and encoded.

    Raises:
        PlanError: ``a``'s shape is not the planned one (oversize included), or
            the transform is unknown.
    """
    shape = plan.shapes[tensor.name]
    if a.shape != shape:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}.shape",
            f"{tensor.name} has shape {a.shape}; the case's is {shape}",
        )
    transform = transform_for(tensor, plan, package=package)
    rows = transform.to_rows(np.asarray(a, dtype=np.float64), plan)
    return encode(rows, _stored_format(tensor, plan.format))


def unpack(
    tensor: TensorDecl, raw: bytes, plan: CasePlan, *, package: str = "ipu_kernels"
) -> Array:
    """The logical values ``tensor``'s region holds (inverse of :func:`pack`).

    An ``accumulator`` output is read in the accumulator format: one vector of
    128 accumulator elements spans ``accumulator_bytes x 128 / row_bytes``
    narrow rows, the stride placement allocated (E06-04).

    Args:
        tensor: The declared tensor.
        raw: The region's bytes, exactly as many as its values take.
        plan: The checked case.
        package: Where family and shared transforms live.

    Returns:
        A float64 array of the planned shape.

    Raises:
        PlanError: ``raw`` is not the region's size, or the transform is unknown.
    """
    shape = plan.shapes[tensor.name]
    transform = transform_for(tensor, plan, package=package)
    stored = _stored_format(tensor, plan.format)
    want = transform.rows(shape) * LANES * FORMATS[stored].element_bytes
    if len(raw) != want:
        raise PlanError(
            plan.manifest,
            f"{tensor.section}.{tensor.name}",
            f"{tensor.name}'s region holds {want} bytes in {stored.value}, given {len(raw)}",
        )
    rows = decode(raw, stored).reshape(-1, LANES)
    return transform.from_rows(rows, plan, shape)
