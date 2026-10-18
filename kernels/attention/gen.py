"""The attention family's case generators (E07-12, E07-13).

Each draws ``RandomState(seed).uniform(-1, 1)`` in FP32, Q before K, with the
seed its case names, so the outputs match the recorded ones bit for bit.
Moving to the harness's seeded generator is a separate change.
"""

from __future__ import annotations

import numpy as np


def _uniform(state: np.random.RandomState, shape: tuple[int, ...]) -> np.ndarray:
    return state.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def legacy_qk(plan, rng):
    """A ``qk_scores`` case: Q and K channel-major ``([n_block,] d, n_tok)``,
    ignoring the harness's ``rng`` (the old seed, the old draw).

    The kernel takes Q staged query-major (``a``, ``[..., n_tok, d]``) and K
    channel-major (``b``, ``[..., d, n_tok]``), so ``y = a @ b``.
    """
    shape = plan.shapes["b"]
    state = np.random.RandomState(plan.params["seed"])
    q = _uniform(state, shape)
    k = _uniform(state, shape)
    return {"a": np.ascontiguousarray(np.swapaxes(q, -1, -2)), "b": k}


def legacy_km(plan, rng):
    """An ``attn_scores_km`` case: the old canonical files hold every head
    (``(n_heads * d, n_tok)``, or ``(p, n_heads, d, n_tok)`` with streams), Q
    then K; the kernel scores the one ``head`` (``a`` and ``b`` are that
    head). The harness's ``rng`` is ignored.

    The kernel takes Q channel-major (``b``, ``[..., d, n_tok]``) and K
    key-major (``a``, ``[..., n_tok, d]``), so ``y = a @ b`` is key-major.
    """
    params = plan.params
    h, d, n = params["head"], params["d"], params["n_tok"]
    state = np.random.RandomState(params["seed"])
    if "p" in params:  # streams: (p, n_heads, d, n_tok), the head of every stream
        shape: tuple[int, ...] = (params["p"], params["n_heads"], d, n)
        q = _uniform(state, shape)[:, h]
        k = _uniform(state, shape)[:, h]
    else:  # (n_heads * d, n_tok), head h's d channel rows
        shape = (params["n_heads"] * d, n)
        q = _uniform(state, shape)[h * d : (h + 1) * d]
        k = _uniform(state, shape)[h * d : (h + 1) * d]
    return {"b": q, "a": np.ascontiguousarray(np.swapaxes(k, -1, -2))}


def legacy_attn_v(plan, rng):
    """An ``attn_v`` case: P ``(n_head, n_tok, n_tok)`` query-major, then V
    ``(n_head, d, n_tok)`` channel-major, as the old cases drew them; the
    golden is ``a @ b^T`` (``matmul_nt``). The harness's ``rng`` is ignored."""
    params = plan.params
    state = np.random.RandomState(params["seed"])
    p = _uniform(state, plan.shapes["b"])
    v = _uniform(state, plan.shapes["a"])
    return {"b": p, "a": v}


def legacy_attn_v_bcast(plan, rng):
    """An ``attn_v_bcast`` case: P and V drawn as for ``attn_v``, then P staged
    key-major (``b[h, s, i] = P[h, i, s]``), so ``y = a @ b``. The harness's
    ``rng`` is ignored."""
    params = plan.params
    state = np.random.RandomState(params["seed"])
    p = _uniform(state, plan.shapes["b"])  # (n_head, n_tok, n_tok): square, query-major
    v = _uniform(state, plan.shapes["a"])
    return {"b": np.ascontiguousarray(p.transpose(0, 2, 1)), "a": v}
