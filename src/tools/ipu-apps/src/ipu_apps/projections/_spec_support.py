"""Shared helpers for the projections kernels' registry declarations.

All twelve projection kernels (proj_{qkv,outproj,ffn1,ffn2}_{144,192,240}_p4)
answer the same query: a fixed input-channel count ``k`` contracted against a
shared weight matrix to produce ``n_out`` output channels, run over a fixed
number of pixel-streams. Each app is a FIXED-SHAPE kernel -- unlike softmax's
"any row count" or "any width", a projection kernel's ``.asm`` is generated
for exactly one (k, n_out) pair, so ``supports`` is an *exact match*, not a
range check.

The query parameters a projection kernel receives are:

``k``        input channels (contraction length)
``n_out``    output channels
``n_streams``  pixel-streams processed per invocation (this family is always
               ``N_STREAM = 4``; kernels outside this family may support 1)

Everything a kernel routes on is these three scalars, so the kernels cannot
disagree about what a given query means. There is deliberately no ``shape``
parameter carrying batch/token dimensions: the token count and token-group
layout are internal to each kernel's fixed .asm and are not something a
caller chooses -- they follow from (k, n_out) picking a specific app.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import OUTPUT, WEIGHT, ShapeBundle

N_STREAM = 4  # pixel-streams every kernel in this family processes per call


@dataclass(frozen=True)
class ProjectionQuery:
    """A projection query reduced to what the kernels route on.

    Attributes:
        k:         Input channels (contraction length).
        n_out:     Output channels.
        n_streams: Pixel-streams requested per invocation.
        bundle:    The shape bundle (weight + output), for reporting.
    """

    k: int
    n_out: int
    n_streams: int
    bundle: ShapeBundle


def projection_query(*, k: int, n_out: int, n_streams: int = N_STREAM) -> ProjectionQuery:
    """Normalise the raw params into the form every projection kernel routes on."""
    bundle = ShapeBundle.of(**{WEIGHT: (int(n_out), int(k))}).with_shapes(
        derived={OUTPUT: (int(n_streams), int(n_out))}
    )
    return ProjectionQuery(
        k=int(k), n_out=int(n_out), n_streams=int(n_streams), bundle=bundle
    )


def positive_dims(q: ProjectionQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.k < 1:
        return f"k ({q.k}) must be >= 1"
    if q.n_out < 1:
        return f"n_out ({q.n_out}) must be >= 1"
    if q.n_streams < 1:
        return f"n_streams ({q.n_streams}) must be >= 1"
    return None


def exact_shape_reason(q: ProjectionQuery, k: int, n_out: int) -> str | None:
    """Refusal reason unless ``q`` matches this kernel's fixed (k, n_out) shape.

    Every projection app is generated for one exact (k, n_out) pair -- there is
    no padding/chunking fallback the way softmax_rows tolerates any row count.
    A mismatch here is always a routing miss, never a partial fit.
    """
    if q.k != k or q.n_out != n_out:
        return (
            f"handles exactly k={k}, n_out={n_out}; this query is "
            f"k={q.k}, n_out={q.n_out}"
        )
    if q.n_streams != N_STREAM:
        return (
            f"handles exactly {N_STREAM} pixel-streams per invocation; "
            f"this query asked for {q.n_streams}"
        )
    return None


FIXED_STREAMS_ONLY = (
    f"Multi-stream (P={N_STREAM}) transformer projection kernel: N_STREAM is "
    f"baked into the .asm and the constructor's input_paths/output_paths "
    f"arity, not a runtime parameter."
)
