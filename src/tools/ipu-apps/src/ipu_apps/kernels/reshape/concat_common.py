"""Shared helpers for the concat kernels' registry declarations.

Concat lives beside ``fold``/``unfold`` in this ``reshape`` family because it
is the same kind of thing: a shape/layout kernel, not an arithmetic op.

Unlike the single-channel-count :func:`~ipu_apps.kernels.reshape.unfold_common.
unfold_query`, concat's query carries TWO channel counts (one per input),
since it concatenates two same-spatial-shape tensors along the channel axis:

``shape``  ``(H, W, C_A, C_B)`` -- spatial height, width, channel count of
           input A, channel count of input B. The output has ``C_A + C_B``
           channels at the same spatial shape.

Each concat kernel here is an exact-shape match (geometry -- row layout,
loop bounds -- is baked into the .asm and harness for one fixed
``(H, W, C_A, C_B)`` tuple), mirroring how ``fold``/``unfold`` route.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes

OP = "concat"

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). This app "
    "copies rows via the FP32 vector path (load -> MULT x1.0 -> ACC.ADD.FIRST "
    "-> ACTIVATE.QUANTIZE identity -> store) and has no narrow (INT8/FP8) "
    "variant."
)


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


def positive_dims(q: ConcatQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.h < 1:
        return f"height ({q.h}) must be >= 1"
    if q.w < 1:
        return f"width ({q.w}) must be >= 1"
    if q.c_a < 1:
        return f"channels_a ({q.c_a}) must be >= 1"
    if q.c_b < 1:
        return f"channels_b ({q.c_b}) must be >= 1"
    return None


def concat_spec(app_class, *, h: int, w: int, c_a: int, c_b: int):
    """KernelSpec for a fixed-``(H, W, C_A, C_B)`` concat kernel.

    An exact-shape match, since C_A/C_B and the loop bounds are baked into
    cr9/cr10 by the harness's setup() for this one tuple.
    """
    name = kernel_folder(app_class)
    dims = (h, w, c_a, c_b)

    def supports(**params):
        q = concat_query(params["shape"])
        bad = positive_dims(q)
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
        op=OP,
        variant=name.removeprefix("concat_"),
        requires=("shape",),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: (
            f"(H, W, C_A, C_B) == {dims} exactly: channel counts and loop "
            f"bounds are fixed constants loaded by setup() for this shape."
        ),
        caveats=lambda **params: (WIDE_VECTOR_ONLY,),
        bundle=lambda **params: concat_query(params["shape"]).bundle,
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
