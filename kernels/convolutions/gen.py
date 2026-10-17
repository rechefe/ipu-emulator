"""Case generators shared by the convolutions' kernels (E07-16).

They keep the ``ipu_apps`` cases' data (``convolutions/cases.py``), so the
migration's FP32 outputs compare bit for bit with the old harness's; moving to
the harness's seeded generator is a separate change.
"""

import numpy as np


def legacy_tiled(plan, rng):
    """x, then w / 4, then b: standard normal FP32 from one ``default_rng(7)``,
    as ``make_cases`` drew them (independent draws since E00-03)."""
    p = plan.params
    k = p["kernel_size"]
    state = np.random.default_rng(7)
    x = state.standard_normal((p["in_channels"], p["height"], p["width"])).astype("<f4")
    w = state.standard_normal((p["out_channels"], p["in_channels"], k, k)).astype("<f4") / 4
    b = state.standard_normal((p["out_channels"],)).astype("<f4")
    return {"x": x, "w": w, "b": b}


def legacy_first_layer(plan, rng):
    """``conv2d_case``'s data for seed 42 with a bias: from ``RandomState(42)``
    the weights (x 0.2), then x (x 0.5), then the bias (x 0.3), in FP32;
    ``filters`` is each filter's bias, then its 27 taps."""
    p = plan.params
    o, c, k = p["out_channels"], p["in_channels"], p["kernel_size"]
    state = np.random.RandomState(42)
    w = (state.randn(o, c, k, k) * 0.2).astype(np.float32)
    x = (state.randn(c, p["height"], p["width"]) * 0.5).astype(np.float32)
    b = (state.randn(o) * 0.3).astype(np.float32)
    return {"x": x, "filters": np.concatenate([b[:, None], w.reshape(o, -1)], axis=1)}
