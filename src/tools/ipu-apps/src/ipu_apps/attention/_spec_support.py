"""Shared helpers for the attention kernels' registry declarations.

The attention family is **four distinct operations**, not four variants of one
op (see the package docstring in ``ipu_apps/attention/__init__.py``): the
query-major chain (``qk_scores`` -> ``attn_v``) and the key-major chain
(``attn_scores_km`` -> ``attn_v_bcast``) produce bit-different results by
design and must never be mixed. Each op therefore gets its own ``op`` value in
every ``SPEC`` in this package.

What the four ops *do* share is vocabulary -- every one of them is indexed by
some subset of ``(n_tok, d, n_head)`` -- so that shape-plumbing lives here once
rather than twelve times:

``n_tok``   tokens per stream (``qk_scores``/``attn_scores_km``: queries ==
            keys; ``attn_v``: queries == keys via P's row count; ``attn_v_bcast``:
            NOT part of the query -- see below)
``d``       head_dim (contraction width for the scores ops, output-channel
            count for the attn@V ops)
``n_head``  attention heads. Only ``attn_scores_km`` exposes a caller-selected
            ``head`` in [0, n_head); the other three ops process every head in
            one kernel invocation, so ``n_head`` there is purely descriptive
            (useful for ``explain``/``bundle``, not a routing constraint).

``attn_v_bcast`` is indexed by ``d`` alone: its three apps (36/48/60) are named
and distinguished purely by head_dim, and N_TOK is a fixed module constant
that is not part of the dirname or a caller-visible parameter -- there is
nothing for a query to assert it against, so it stays out of the query
envelope for this op (each app's ``explain``/``bundle`` can still report its
own fixed N_TOK for information).
"""

from __future__ import annotations

from dataclasses import dataclass


def positive_dims(**dims: int) -> str | None:
    """Return a refusal reason if any named extent is non-positive.

    Shared by all four ops' ``supports`` callbacks so "n_tok must be >= 1"
    style refusals read identically everywhere instead of being retyped with
    slightly different wording per app.
    """
    for name, value in dims.items():
        if value < 1:
            return f"{name} ({value}) must be >= 1"
    return None


@dataclass(frozen=True)
class ScoresQuery:
    """Query shape for ``qk_scores`` and ``attn_scores_km``: n_tok x d.

    Both ops compute an N_TOK x N_TOK score matrix contracted over d
    (head_dim); a kernel matches only when both extents match its fixed
    module constants exactly (no padding, no chunking across apps).
    """

    n_tok: int
    d: int


def scores_query(*, n_tok: int, d: int) -> ScoresQuery:
    return ScoresQuery(n_tok=int(n_tok), d=int(d))


@dataclass(frozen=True)
class AttnVQuery:
    """Query shape for ``attn_v``: n_tok x d (same vocabulary as ScoresQuery,
    kept as a distinct type because the two ops are never interchangeable and
    a routing bug that accidentally matched a scores query against an attn_v
    kernel should be a type error, not a coincidence of field names).
    """

    n_tok: int
    d: int


def attn_v_query(*, n_tok: int, d: int) -> AttnVQuery:
    return AttnVQuery(n_tok=int(n_tok), d=int(d))


@dataclass(frozen=True)
class AttnVBcastQuery:
    """Query shape for ``attn_v_bcast``: d alone (see module docstring)."""

    d: int


def attn_v_bcast_query(*, d: int) -> AttnVBcastQuery:
    return AttnVBcastQuery(d=int(d))
