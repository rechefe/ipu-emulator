"""Shared code for the attention kernels: chain vocabulary and SPEC helper.

Twelve kernels implement two independent QK^T / attn@V chains, each available
for all three MobileViT layers (L3/L4/L5):

- **query-major:** ``qk_scores_*`` -> ``attn_v_*`` (attn@V via AGG)
- **key-major:**   ``attn_scores_km_*`` -> ``attn_v_bcast_*`` (attn@V via
  broadcast ACC.ADD, no AGG)

The two chains produce bit-different results by design and must never be
mixed -- see ``docs/content/kernels/mobilevit.md``'s chaining rules.

**Four operations, not four variants of one op.** Each stage gets its own
``op`` value in every ``SPEC`` (:data:`CHAINS`), and the registry routes on
``op`` before anything else. That is what keeps the chains apart: a caller
holding query-major scores asks for ``attn_v`` and can only ever be handed an
AGG kernel, whose P input is query-major; a caller holding key-major scores
asks for ``attn_v_bcast`` and can only ever be handed a broadcast kernel. No
shape parameter could do that job -- both attn@V kernels of a layer take the
same ``(n_tok, d)`` -- and every kernel is also tagged with its chain.

What the four ops *do* share is vocabulary -- every one of them is indexed by
some subset of ``(n_tok, d, head)`` -- so that shape-plumbing lives here once
rather than twelve times:

``n_tok``   tokens per stream (``qk_scores``/``attn_scores_km``: queries ==
            keys; ``attn_v``: queries == keys via P's row count). Optional for
            ``attn_v_bcast`` -- see below.
``d``       head_dim (contraction width for the scores ops, output-channel
            count for the attn@V ops)
``head``    ``attn_scores_km`` only, optional (default 0): which head of the
            multi-head input file to score. The other three ops process every
            head in one kernel invocation.

``attn_v_bcast`` is indexed by ``d``: its three kernels (36/48/60) are named
and distinguished by head_dim, and N_TOK is a fixed module constant, not part
of the dirname or a constructor parameter. A query may still carry ``n_tok``
(e.g. the same ``(n_tok, d)`` it used for ``attn_scores_km``), and then it must
match that fixed N_TOK -- otherwise ``d=36`` alone would route a 16-token
problem to the 256-token kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

from ipu_apps.kernel_registry import ExecutionConfig, folder_spec, kernel_folder, no, yes

QUERY_MAJOR = "query-major"
KEY_MAJOR = "key-major"

# chain -> (scores op, attn@V op). A kernel's op determines its chain.
CHAINS = {
    QUERY_MAJOR: ("qk_scores", "attn_v"),
    KEY_MAJOR: ("attn_scores_km", "attn_v_bcast"),
}
CHAIN_OF = {op: chain for chain, ops in CHAINS.items() for op in ops}


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
    """Query shape for ``attn_v_bcast``: d, plus n_tok only if the query
    carries it (see module docstring)."""

    d: int
    n_tok: int | None = None


def attn_v_bcast_query(*, d: int, n_tok: int | None = None) -> AttnVBcastQuery:
    return AttnVBcastQuery(d=int(d), n_tok=None if n_tok is None else int(n_tok))


def attention_spec(app_class, *, op: str, n_tok: int, d: int, explain: str,
                   bundle, heads: int | None = None):
    """KernelSpec for a fixed ``(n_tok, d)`` attention kernel of operation ``op``.

    ``supports`` is the single source of truth for the kernel's exact-match
    domain -- no padding, no chunking, so the cheapest possible claim (cost 0).
    ``heads`` is given for the ``attn_scores_km`` kernels, which score one
    selected head of an ``heads``-head input file: the optional ``head`` query
    parameter must lie in ``[0, heads)`` and reaches the constructor through
    ``build``. ``explain`` is the fixed explanation string; ``bundle`` maps the
    query to its :class:`~ipu_apps.kernel_registry.ShapeBundle`.
    """
    if op not in CHAIN_OF:
        raise ValueError(f"unknown attention op {op!r}; expected one of {sorted(CHAIN_OF)}")
    name = kernel_folder(app_class)
    bcast = op == "attn_v_bcast"

    def supports(**params):
        if bcast:
            q = attn_v_bcast_query(d=params["d"], n_tok=params.get("n_tok"))
            dims = {"d": q.d} if q.n_tok is None else {"n_tok": q.n_tok, "d": q.d}
        else:
            make = attn_v_query if op == "attn_v" else scores_query
            q = make(n_tok=params["n_tok"], d=params["d"])
            dims = {"n_tok": q.n_tok, "d": q.d}
        bad = positive_dims(**dims)
        if bad:
            return no(bad)
        if "n_tok" in dims and dims["n_tok"] != n_tok:
            what = "is fixed at" if bcast else "handles exactly"
            return no(f"{what} n_tok={n_tok}; got {dims['n_tok']}")
        if dims["d"] != d:
            return no(f"handles exactly d={d}; got {dims['d']}")
        if heads is not None:
            head = params.get("head", 0)
            if type(head) is not int or not 0 <= head < heads:
                return no(f"head must be an integer in [0, {heads}); got {head!r}")
        return yes()

    return folder_spec(
        app_class,
        op=op,
        variant=name.rpartition("_")[2],
        requires=("d",) if bcast else ("n_tok", "d"),
        tags=("fp32-wide", CHAIN_OF[op]),
        supports=supports,
        build=(lambda **params: {"head": params.get("head", 0)}) if heads is not None
        else (lambda **params: {}),
        explain=lambda **params: explain,
        bundle=bundle,
        # Exact-shape match: no padding, no chunking. Cheapest possible claim.
        cost=lambda **params: 0.0,
        execution=ExecutionConfig(mode="fp32"),
    )
