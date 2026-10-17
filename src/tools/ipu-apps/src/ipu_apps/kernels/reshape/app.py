"""Shared code for the reshape kernels: unfold/fold query vocabulary and SPEC helpers.

Unfold and fold are shape/layout kernels, not arithmetic ops. Every
one of them is written against exactly one shape (the geometry -- stripe
counts, packing order, loop bounds, register layout -- is baked into the .asm
and harness), so each ``supports`` is an exact-shape match rather than a
bound, and there is no flattening or axis convention to normalise the way
softmax's ``dim`` needs. The query helpers exist so the kernels of one op
cannot disagree about how a ``shape`` parameter unpacks. ``depth_to_space``
and ``identity`` keep their own memory-layout specs and use none of this; the
concat kernels moved to ``kernels/reshape`` (E07-04).

**Unfold / fold.** The three unfold kernels and the three fold kernels that
invert them take one parameter:

``shape``  the input spatial shape, ``(H, W, C)``
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import (
    ExecutionConfig,
    ShapeBundle,
    folder_spec,
    kernel_folder,
    no,
    yes,
)
from ipu_isa.formats import ElementFormat

UNFOLD_WIDE_VECTOR_ONLY = (
    "FP32 element format only (fmt=ElementFormat.FP32). These apps "
    "rearrange spatial data via ACC.STRIDE (unfold) or ACC.RESHAPE (fold) over "
    "the FP32 vector path and have no narrow (INT8/FP8) variant."
)

def positive_dims(**dims: int) -> str | None:
    """Return a refusal reason naming the first non-positive extent."""
    for name, value in dims.items():
        if value < 1:
            return f"{name} ({value}) must be >= 1"
    return None


# -- unfold / fold ------------------------------------------------------------


@dataclass(frozen=True)
class UnfoldQuery:
    """An unfold or fold query reduced to what the kernels route on.

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


def unfold_spec(
    app_class,
    *,
    op: str,
    h: int,
    w: int,
    c: int,
    geometry: str = "stripe count, spatial row packing, register layout",
):
    """KernelSpec for a fixed-``(H, W, C)`` unfold or fold kernel.

    ``supports`` is an exact-shape match, since the stripe/packing geometry is
    baked into the .asm for this one (H, W, C) triple. ``geometry`` names what
    is fixed, for ``explain``.
    """
    name = kernel_folder(app_class)

    def supports(**params):
        q = unfold_query(params["shape"])
        bad = positive_dims(height=q.h, width=q.w, channels=q.c)
        if bad:
            return no(bad)
        if (q.h, q.w, q.c) != (h, w, c):
            return no(f"handles exactly (H, W, C) = ({h}, {w}, {c}); got ({q.h}, {q.w}, {q.c})")
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
        caveats=lambda **params: (UNFOLD_WIDE_VECTOR_ONLY,),
        bundle=lambda **params: unfold_query(params["shape"]).bundle,
        # Exact-shape match: no two kernels of one op share a triple.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(fmt=ElementFormat.FP32),
    )

