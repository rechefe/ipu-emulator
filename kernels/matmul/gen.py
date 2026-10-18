"""The matmul cases' data (E07-07, E07-08), shared by the family.

``seeded_uniform`` draws from ``RandomState(SEED)``, A ``(tokens, k)`` then W
``(n, k)``, uniform in ``[-1, 1)``, FP32, so the outputs match the recorded
ones bit for bit. A template lays them out as its tensors take them.
"""

import numpy as np

#: The default case's seed.
SEED = 0


def _draw(tokens, k, n):
    state = np.random.RandomState(SEED)
    a = state.uniform(-1.0, 1.0, size=(tokens, k)).astype(np.float32)
    w = state.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    return a, w


def seeded_uniform(plan, rng):
    """Row-major kernels: ``a`` is A ``(m, k)``, ``b`` is W^T ``(k, n)``."""
    m, k = plan.shapes["a"]
    n = plan.shapes["b"][1]
    a, w = _draw(m, k, n)
    return {"a": a, "b": np.ascontiguousarray(w.T)}


def seeded_uniform_channel_major(plan, rng):
    """The ``*_x128`` kernels: ``b`` is the data D = A^T, ``(n_tg, k, n_tok)``
    (token ``t`` of A is token ``t % n_tok`` of group ``t // n_tok``), and ``a``
    is W ``(n, k)``."""
    n_tg, k, n_tok = plan.shapes["b"]
    n = plan.shapes["a"][0]
    a, w = _draw(n_tg * n_tok, k, n)
    return {"a": w, "b": np.ascontiguousarray(a.reshape(n_tg, n_tok, k).transpose(0, 2, 1))}
