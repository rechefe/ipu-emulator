"""Shared helpers for the unfold kernels' registry declarations.

All three unfold kernels answer the same query -- a fixed ``(H, W, C)`` spatial
shape -- so the parameter unpacking and the shared refusal live here rather
than being repeated three times.

The query parameter an unfold kernel receives is:

``shape``  the input spatial shape, as ``(H, W, C)``

Each kernel is written against exactly one ``(H, W, C)`` triple (the geometry
is baked into the .asm -- stripe counts, packing order, register layout), so
there is no flattening or axis convention to normalise here the way softmax's
``dim`` needs; :func:`unfold_query` exists so the three kernels cannot disagree
about how a ``shape`` parameter unpacks into ``h, w, c``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ShapeBundle

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "rearrange spatial data via ACC.STRIDE over the FP32 vector path and have "
    "no narrow (INT8/FP8) variant."
)


@dataclass(frozen=True)
class UnfoldQuery:
    """An unfold query reduced to what the kernels route on.

    Attributes:
        h: Spatial height.
        w: Spatial width.
        c: Channel count.
        bundle: The shape bundle for this query.
    """

    h: int
    w: int
    c: int
    bundle: ShapeBundle


def unfold_query(shape) -> UnfoldQuery:
    """Normalise a ``shape=(H, W, C)`` parameter into what kernels route on."""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 3:
        raise ValueError(f"unfold shape must be rank 3 (H, W, C); got {dims}")
    h, w, c = dims
    bundle = ShapeBundle.of(input=dims).with_shapes(derived={"output": dims})
    return UnfoldQuery(h=h, w=w, c=c, bundle=bundle)


def positive_dims(q: UnfoldQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.h < 1:
        return f"height ({q.h}) must be >= 1"
    if q.w < 1:
        return f"width ({q.w}) must be >= 1"
    if q.c < 1:
        return f"channels ({q.c}) must be >= 1"
    return None
