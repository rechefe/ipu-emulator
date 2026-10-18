"""The softmax family's case generators (E07-11).

``legacy_randn`` keeps the old cases' data (``softmax/cases.py``,
``random_case``): ``RandomState(seed).randn(rows, cols) * scale`` in FP32, so
the migration's outputs compare bit for bit with the old harness's. ``cols``
is the kernel's row length: ``n`` for the row kernels, ``width`` for the
column kernels. Moving to the harness's seeded generator is a separate change.
"""

from __future__ import annotations

import numpy as np


def legacy_randn(plan, rng):
    """x from ``RandomState(seed)``, ignoring the harness's ``rng`` (the old seed, the old draw)."""
    rows, cols = plan.shapes["x"]
    x = np.random.RandomState(plan.params["seed"]).randn(rows, cols) * plan.params["scale"]
    return {"x": x.astype(np.float32)}


def constant(plan, rng):
    """x with ``scale`` in every element: softmax is then 1 / cols everywhere."""
    return {"x": np.full(plan.shapes["x"], plan.params["scale"], dtype=np.float32)}
