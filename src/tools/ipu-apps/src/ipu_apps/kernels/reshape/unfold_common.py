"""Shared helpers for the unfold and fold kernels' registry declarations.

All three unfold kernels (and the three fold kernels that invert them) answer
the same query -- a fixed ``(H, W, C)`` spatial shape -- so the parameter
unpacking and the shared refusal live here rather than being repeated six
times.

The query parameter an unfold/fold kernel receives is:

``shape``  the input spatial shape, as ``(H, W, C)``

Each kernel is written against exactly one ``(H, W, C)`` triple (the geometry
is baked into the .asm -- stripe counts, packing order, register layout), so
there is no flattening or axis convention to normalise here the way softmax's
``dim`` needs; :func:`unfold_query` exists so the kernels cannot disagree
about how a ``shape`` parameter unpacks into ``h, w, c``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "rearrange spatial data via ACC.STRIDE (unfold) or ACC.RESHAPE (fold) over "
    "the FP32 vector path and have "
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


def unfold_spec(app_class, *, op: str, h: int, w: int, c: int,
                geometry: str = "stripe count, spatial row packing, register layout"):
    """KernelSpec for a fixed-``(H, W, C)`` unfold or fold kernel.

    `supports` is the single source of truth for the kernel's domain: it is an
    exact-shape match, since the stripe/packing geometry is baked into the
    .asm for this one (H, W, C) triple. ``geometry`` names what is fixed, for
    ``explain``.
    """
    name = kernel_folder(app_class)

    def supports(**params):
        q = unfold_query(params["shape"])
        bad = positive_dims(q)
        if bad:
            return no(bad)
        if (q.h, q.w, q.c) != (h, w, c):
            return no(
                f"handles exactly (H, W, C) = ({h}, {w}, {c}); got ({q.h}, {q.w}, {q.c})"
            )
        return yes()

    return folder_spec(
        app_class,
        op=op,
        variant=name.partition("_")[2],
        requires=("shape",),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: (
            f"(H, W, C) == ({h}, {w}, {c}) exactly: geometry ({geometry}) "
            f"is fixed in the .asm for this shape."
        ),
        caveats=lambda **params: (WIDE_VECTOR_ONLY,),
        bundle=lambda **params: unfold_query(params["shape"]).bundle,
        # Exact-shape match: no two kernels of one op share a triple.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
