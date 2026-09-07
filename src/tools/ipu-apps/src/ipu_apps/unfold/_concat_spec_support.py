"""Shared helpers for the concat kernels' registry declarations.

Concat lives beside ``fold``/``unfold`` in this ``unfold/`` package because it
is the same kind of thing: a shape/layout kernel, not an arithmetic op. It has
no reason to import from (or depend on) ``convolutions_universal`` -- that
family lives on unrelated branch ancestry that is not available here.

Unlike the single-channel-count ``unfold_query`` in :mod:`ipu_apps.unfold.
_spec_support`, concat's query carries TWO channel counts (one per input),
since it concatenates two same-spatial-shape tensors along the channel axis:

``shape``  ``(H, W, C_A, C_B)`` -- spatial height, width, channel count of
           input A, channel count of input B. The output has ``C_A + C_B``
           channels at the same spatial shape.

Each concat kernel here is an exact-shape match (geometry -- row layout,
loop bounds -- is baked into the .asm for one fixed ``(H, W, C_A, C_B)``
tuple), mirroring how ``fold``/``unfold``/``residual_add`` all route.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ShapeBundle

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
