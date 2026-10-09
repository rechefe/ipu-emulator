"""Shared code for the normalize kernels: layernorm query vocabulary and SPEC helper.

All four layernorm kernels answer the same query -- a channel count and a
token count -- so the parameter unpacking and the constants they reason about
live here rather than being repeated four times. ``l2_normalize_channels``
keeps its own memory-layout spec and does not use any of this.

The query parameters a layernorm kernel receives are:

``shape``  the input shape, ``(channels, tokens)``, matching the on-disk and
           in-XMEM layout: one channel per row, tokens in the lanes. This is
           deliberately *not* torch's ``(..., normalized_shape)`` convention
           (tokens, channels) -- the reduction axis here is always channels,
           independently per token, and every kernel's docstring writes the
           computation as ``output[ch, i]`` over that same layout.

Everything a kernel actually routes on -- ``channels``, ``tokens`` -- is
derived from ``shape`` by :func:`layernorm_query`, so the kernels cannot
disagree about what a given shape means. Each of the four kernels here is an
exact-shape match (one (channels, tokens) pair per app, not a range), so
``supports`` is a single equality check rather than a bound; see
:func:`layernorm_spec`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import (
    ExecutionConfig, ShapeBundle, Table, folder_spec, kernel_folder, no, yes,
)

OP = "layernorm"
LANES = 128  # datapath width; not itself a routing constraint here

WIDE_VECTOR_ONLY = (
    "Wide-vector FP32 mode only (SPEC.execution mode='fp32'). These apps "
    "build on rsqrt over an FP32 vector path and have no narrow (INT8/FP8) "
    "variant."
)


@dataclass(frozen=True)
class LayerNormQuery:
    """A layernorm query reduced to what the kernels route on.

    Attributes:
        channels: Rows of the (channels, tokens) problem -- the reduction
            axis (normalization is over channels, independently per token).
        tokens:   Columns of that problem.
        bundle:   The shape bundle.
    """

    channels: int
    tokens: int
    bundle: ShapeBundle


def layernorm_query(shape) -> LayerNormQuery:
    """Normalise ``shape`` into the form every layernorm kernel routes on."""
    dims = tuple(int(d) for d in shape)
    if len(dims) != 2:
        raise ValueError(
            f"layernorm shape must be rank-2 (channels, tokens); got {dims}"
        )
    channels, tokens = dims
    bundle = ShapeBundle.of(input=dims).with_shapes(derived={"output": dims})
    return LayerNormQuery(channels=channels, tokens=tokens, bundle=bundle)


def positive_dims(q: LayerNormQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.channels < 1:
        return f"channels ({q.channels}) must be >= 1"
    if q.tokens < 1:
        return f"tokens ({q.tokens}) must be >= 1"
    return None


def layernorm_tables(channels: int, n_tg: int) -> dict[str, Table]:
    """XMEM tables of the fixed-shape layernorm kernels (see ``ipu_emu.table_trace``).

    The input holds one row per (channel, token group), groups interleaved.
    Each token group runs the whole pipeline over its rows with stride
    ``n_tg``: x is read twice (mean, then centring), gamma/beta once, and
    the centred/normalised scratch -- one row per channel, reused by every
    group -- is written and re-read over four passes, each row stored one
    cycle after the next is loaded (one row back). The x and scratch load
    streams read one step past their last row; those rows are allocated as
    overhead. The one-row constants and per-group scalars (mean, sum of
    squares, 1/sigma) stay in their own rows.
    """
    affine_rows = -(-channels // LANES)
    return {
        "data": Table(cr=13, repetitions=2 * n_tg, rows=channels * n_tg, overhead_rows=n_tg),
        "gamma": Table(cr=11, repetitions=n_tg, rows=affine_rows),
        "beta": Table(cr=2, repetitions=n_tg, rows=affine_rows),
        "ones": Table(cr=3, rows=1),
        "neg_inv_n": Table(cr=4, rows=1),
        "inv_n": Table(cr=5, rows=1),
        "neg_mean": Table(cr=6, rows=1),
        "centered": Table(cr=7, jump_back=1, repetitions=4 * n_tg, rows=channels,
                          overhead_rows=1),
        "temp": Table(cr=8, rows=1),
        "invstd": Table(cr=9, rows=1),
        "output": Table(cr=10, repetitions=n_tg, rows=channels * n_tg),
    }


def layernorm_spec(app_class, *, channels: int, tokens: int, tables=None):
    """KernelSpec for a fixed ``(channels, tokens)`` layernorm kernel.

    ``tokens`` is the kernel's *total* token count (all token groups).
    ``tables`` is the kernel's ``SPEC.tables`` declaration, when it has one.
    """
    name = kernel_folder(app_class)

    def supports(**params):
        q = layernorm_query(params["shape"])
        bad = positive_dims(q)
        if bad:
            return no(bad)
        if (q.channels, q.tokens) != (channels, tokens):
            return no(
                f"handles exactly (channels={channels}, tokens={tokens}); "
                f"this query is (channels={q.channels}, tokens={q.tokens})"
            )
        return yes()

    return folder_spec(
        app_class,
        op=OP,
        variant=name.removeprefix("layernorm_"),
        requires=("shape",),
        tags=("fp32-wide",),
        supports=supports,
        build=lambda **params: {},
        explain=lambda **params: f"exact match: (channels={channels}, tokens={tokens}).",
        caveats=lambda **params: (WIDE_VECTOR_ONLY,),
        bundle=lambda **params: layernorm_query(params["shape"]).bundle,
        # Exact-shape match: no padding, no chunking. Cheapest possible claim.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
        tables=tables,
    )
