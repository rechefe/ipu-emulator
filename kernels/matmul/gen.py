"""The ``ipu_apps`` matmul cases' data (E07-07, E07-08), shared by the family.

``legacy_uniform`` draws as the old ``matmul/cases.py`` did: from
``RandomState(seed)``, A ``(tokens, k)`` then W ``(n, k)``, uniform in
``[-1, 1)``, FP32, so the migration's outputs compare bit for bit with the old
harness's. A template lays them out as its tensors take them.
"""

import numpy as np

#: The ``ipu_apps`` default case's seed.
SEED = 0


def _draw(tokens, k, n):
    state = np.random.RandomState(SEED)
    a = state.uniform(-1.0, 1.0, size=(tokens, k)).astype(np.float32)
    w = state.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    return a, w


def legacy_uniform(plan, rng):
    """Row-major kernels: ``a`` is A ``(m, k)``, ``b`` is W^T ``(k, n)``."""
    m, k = plan.shapes["a"]
    n = plan.shapes["b"][1]
    a, w = _draw(m, k, n)
    return {"a": a, "b": np.ascontiguousarray(w.T)}
