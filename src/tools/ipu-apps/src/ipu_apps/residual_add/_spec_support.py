"""Shared helpers for the residual-add kernels' registry declarations.

All three residual-add kernels answer the same query -- a token count and a
channel count -- so the parameter unpacking and the constants they reason
about live here rather than being repeated three times.

The query parameters a residual-add kernel receives are:

``shape``  the input shape, ``(tokens, channels)`` -- torch's elementwise
           convention, batch/sequence dim before the feature dim. A and B
           share this shape (residual add is elementwise between two
           identically-shaped tensors); the output shape equals it.

Everything a kernel actually routes on -- ``tokens``, ``channels`` -- is
derived from ``shape`` by :func:`residual_add_query`. Each of the three
kernels here is an exact-shape match (one (tokens, channels) pair per app,
not a range), so ``supports`` is a single equality check rather than a bound.

Two of the three kernels (``residual_add_16x240``, ``residual_add_64x192``)
give each channel a whole 128-lane XMEM row and crop the unused tail lanes at
teardown; the third (``residual_add_256x144``) has no channel-per-row
structure at all -- ``tokens > LANES`` there, so the (tokens, channels)
problem is simply flattened into ``ceil(tokens / LANES) * channels`` full
128-lane rows with no padding to crop. Routing only needs the exact
(tokens, channels) pair, not the row layout, so that difference stays inside
each kernel's own harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ShapeBundle

LANES = 128  # datapath width; not itself a routing constraint here

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 debug mode only (wide_vector_debug=True). These apps "
    "add over an FP32 vector path and have no narrow (INT8/FP8) variant."
)


@dataclass(frozen=True)
class ResidualAddQuery:
    """A residual-add query reduced to what the kernels route on.

    Attributes:
        tokens:   Rows of the (tokens, channels) problem.
        channels: Columns of that problem.
        bundle:   The shape bundle.
    """

    tokens: int
    channels: int
    bundle: ShapeBundle


def residual_add_query(shape) -> ResidualAddQuery:
    """Normalise ``shape`` into the form every residual-add kernel routes on."""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 2:
        raise ValueError(
            f"residual_add shape must be rank-2 (tokens, channels); got {dims}"
        )
    tokens, channels = dims
    bundle = ShapeBundle.of(input=dims).with_shapes(derived={"output": dims})
    return ResidualAddQuery(tokens=tokens, channels=channels, bundle=bundle)


def positive_dims(q: ResidualAddQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.tokens < 1:
        return f"tokens ({q.tokens}) must be >= 1"
    if q.channels < 1:
        return f"channels ({q.channels}) must be >= 1"
    return None
