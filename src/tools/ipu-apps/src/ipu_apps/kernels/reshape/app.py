"""Shared code for the reshape kernels: unfold/fold and concat query vocabulary and SPEC helpers.

Unfold, fold and concat are shape/layout kernels, not arithmetic ops. Every
one of them is written against exactly one shape (the geometry -- stripe
counts, packing order, loop bounds, register layout -- is baked into the .asm
and harness), so each ``supports`` is an exact-shape match rather than a
bound, and there is no flattening or axis convention to normalise the way
softmax's ``dim`` needs. The query helpers exist so the kernels of one op
cannot disagree about how a ``shape`` parameter unpacks. ``depth_to_space``
and ``identity`` keep their own memory-layout specs and use none of this.

**Unfold / fold.** The three unfold kernels and the three fold kernels that
invert them take one parameter:

``shape``  the input spatial shape, ``(H, W, C)``

**Concat.** Concat joins two same-spatial-shape tensors along the channel
axis, so its query carries two channel counts:

``shape``  ``(H, W, C_A, C_B)`` -- spatial height, width, channel count of
           input A, channel count of input B. The output has ``C_A + C_B``
           channels at the same spatial shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes

CONCAT_OP = "concat"

UNFOLD_WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "rearrange spatial data via ACC.STRIDE (unfold) or ACC.RESHAPE (fold) over "
    "the FP32 vector path and have no narrow (INT8/FP8) variant."
)

CONCAT_WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). This app "
    "copies rows via the FP32 vector path (load -> MULT x1.0 -> ACC.ADD.FIRST "
    "-> ACTIVATE.QUANTIZE identity -> store) and has no narrow (INT8/FP8) "
    "variant."
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


def unfold_spec(app_class, *, op: str, h: int, w: int, c: int,
                geometry: str = "stripe count, spatial row packing, register layout"):
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
        caveats=lambda **params: (UNFOLD_WIDE_VECTOR_ONLY,),
        bundle=lambda **params: unfold_query(params["shape"]).bundle,
        # Exact-shape match: no two kernels of one op share a triple.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )


# -- concat -------------------------------------------------------------------


@dataclass(frozen=True)
class ConcatQuery:
    """A concat query reduced to what the kernels route on.

    Attributes:
        h: Spatial height (shared by both inputs and the output).
        w: Spatial width (shared by both inputs and the output).
        c_a: Channel count of input A.
        c_b: Channel count of input B.
        bundle: The shape bundle for this query.
    """

    h: int
    w: int
    c_a: int
    c_b: int
    bundle: ShapeBundle


def concat_query(shape) -> ConcatQuery:
    """Normalise a ``shape=(H, W, C_A, C_B)`` parameter into what kernels route on."""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 4:
        raise ValueError(
            f"concat shape must be rank 4 (H, W, C_A, C_B); got {dims}"
        )
    h, w, c_a, c_b = dims
    bundle = ShapeBundle.of(input_a=(h, w, c_a), input_b=(h, w, c_b)).with_shapes(
        derived={"output": (h, w, c_a + c_b)}
    )
    return ConcatQuery(h=h, w=w, c_a=c_a, c_b=c_b, bundle=bundle)


def concat_spec(app_class, *, h: int, w: int, c_a: int, c_b: int):
    """KernelSpec for a fixed-``(H, W, C_A, C_B)`` concat kernel.

    An exact-shape match, since C_A/C_B and the loop bounds are baked into
    CR9/CR10 by the harness's setup() for this one tuple.
    """
    name = kernel_folder(app_class)
    dims = (h, w, c_a, c_b)

    def supports(**params):
        q = concat_query(params["shape"])
        bad = positive_dims(height=q.h, width=q.w, channels_a=q.c_a, channels_b=q.c_b)
        if bad:
            return no(bad)
        if (q.h, q.w, q.c_a, q.c_b) != dims:
            return no(
                f"handles exactly (H, W, C_A, C_B) = {dims}; "
                f"got ({q.h}, {q.w}, {q.c_a}, {q.c_b})"
            )
        return yes()

    return folder_spec(
        app_class,
        op=CONCAT_OP,
        variant=name.removeprefix("concat_"),
        requires=("shape",),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: (
            f"(H, W, C_A, C_B) == {dims} exactly: channel counts and loop "
            f"bounds are fixed constants loaded by setup() for this shape."
        ),
        caveats=lambda **params: (CONCAT_WIDE_VECTOR_ONLY,),
        bundle=lambda **params: concat_query(params["shape"]).bundle,
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
