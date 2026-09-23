"""Shared code for the elementwise kernels: residual-add query vocabulary and SPEC helper.

All three fixed-shape residual-add kernels (``residual_add_16x240``,
``residual_add_64x192``, ``residual_add_256x144``) answer the same query -- a
token count and a channel count -- so the parameter unpacking and the
constants they reason about live here rather than being repeated three times.

The query parameters a residual-add kernel receives are:

``shape``  the input shape, ``(tokens, channels)`` -- torch's elementwise
           convention, batch/sequence dim before the feature dim. A and B
           share this shape (residual add is elementwise between two
           identically-shaped tensors); the output shape equals it.

Everything a kernel actually routes on -- ``tokens``, ``channels`` -- is
derived from ``shape`` by :func:`residual_add_query`. Each of the three
kernels here is an exact-shape match (one (tokens, channels) pair per app,
not a range), so ``supports`` is a single equality check rather than a bound;
see :func:`residual_add_spec`.

Two of the three kernels (``residual_add_16x240``, ``residual_add_64x192``)
give each channel a whole 128-lane XMEM row and crop the unused tail lanes at
teardown (16x240) or leave them to the consumer (64x192); the third
(``residual_add_256x144``) has no channel-per-row structure at all --
``tokens > LANES`` there, so the (tokens, channels) problem is simply
flattened into ``ceil(tokens / LANES) * channels`` full 128-lane rows with no
padding to crop. Routing only needs the exact (tokens, channels) pair, not the
row layout, so that difference stays inside each kernel's own harness.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import (
    ExecutionConfig, ShapeBundle, folder_spec, kernel_folder, no, yes,
)

RESIDUAL_ADD_OP = "residual_add"
LANES = 128  # datapath width; not itself a routing constraint here

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 mode only (SPEC.execution mode='fp32'). These apps "
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


def residual_add_spec(app_class, *, tokens: int, channels: int):
    """KernelSpec for a fixed ``(tokens, channels)`` residual-add kernel."""
    name = kernel_folder(app_class)

    def supports(**params):
        q = residual_add_query(params["shape"])
        bad = positive_dims(q)
        if bad:
            return no(bad)
        if (q.tokens, q.channels) != (tokens, channels):
            return no(
                f"handles exactly (tokens={tokens}, channels={channels}); "
                f"this query is (tokens={q.tokens}, channels={q.channels})"
            )
        return yes()

    return folder_spec(
        app_class,
        op=RESIDUAL_ADD_OP,
        variant=name.removeprefix("residual_add_"),
        requires=("shape",),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: f"exact match: (tokens={tokens}, channels={channels}).",
        caveats=lambda **params: (WIDE_VECTOR_ONLY,),
        bundle=lambda **params: residual_add_query(params["shape"]).bundle,
        # Exact-shape match: no padding, no chunking. Cheapest possible claim.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
