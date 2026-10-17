"""The attention family's case generators (E07-12).

They keep the ``ipu_apps`` cases' data (``attention/cases.py`` and each
kernel's ``cases.py``): ``RandomState(seed).uniform(-1, 1)`` in FP32, Q drawn
before K, so the migration's outputs compare bit for bit with the old
harness's. Moving to the harness's seeded generator is a separate change.
"""

from __future__ import annotations

import numpy as np


def _uniform(state: np.random.RandomState, shape: tuple[int, ...]) -> np.ndarray:
    return state.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def legacy_qk(plan, rng):
    """A ``qk_scores`` case: Q and K channel-major ``([n_block,] d, n_tok)``.

    The kernel takes Q staged query-major (``a``, ``[..., n_tok, d]``) and K
    channel-major (``b``, ``[..., d, n_tok]``), so ``y = a @ b``.
    """
    p = plan.params
    shape = (*plan.shapes["b"][:-2], p["d"], p["n_tok"])
    state = np.random.RandomState(p["seed"])
    q = _uniform(state, shape)
    k = _uniform(state, shape)
    return {"a": np.ascontiguousarray(np.swapaxes(q, -1, -2)), "b": k}


def legacy_km(plan, rng):
    """An ``attn_scores_km`` case: the old canonical files hold every head
    (``(n_heads * d, n_tok)``, or ``(p, n_heads, d, n_tok)`` with streams), Q
    then K; the kernel scores the one ``head``.

    The kernel takes Q channel-major (``b``, ``[..., d, n_tok]``) and K
    key-major (``a``, ``[..., n_tok, d]``), so ``y = a @ b`` is key-major.
    """
    p = plan.params
    h, d, n = p["head"], p["d"], p["n_tok"]
    state = np.random.RandomState(p["seed"])
    if "p" in p:  # streams: (p, n_heads, d, n_tok), the head of every stream
        shape: tuple[int, ...] = (p["p"], p["n_heads"], d, n)
        q = _uniform(state, shape)[:, h]
        k = _uniform(state, shape)[:, h]
    else:  # (n_heads * d, n_tok), head h's d channel rows
        shape = (p["n_heads"] * d, n)
        q = _uniform(state, shape)[h * d : (h + 1) * d]
        k = _uniform(state, shape)[h * d : (h + 1) * d]
    return {"b": q, "a": np.ascontiguousarray(np.swapaxes(k, -1, -2))}
