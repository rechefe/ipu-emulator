"""The pointwise and depthwise cases' ipu_apps data (E07-18), for the proof only.

generator = "../gen.py:<fn>": convolutions/cases.py's _data and its
data modes, seed 0 and draw order (weights randn * 0.3 (1x1) or 0.2
(3x3), then the input randn * 0.5, then the bias randn * 0.3 when the
old case had one), so the outputs compare bit for bit with the old harness's.
The next commit moves the cases to the harness's seeded generator and deletes
this file.
"""

import numpy as np


def _draw(plan, *, out_ch, group_in, k, with_bias, mode):
    p = plan.params
    r = np.random.RandomState(p.get("seed", 0))
    w = (r.randn(out_ch, group_in, k, k) * (0.2 if k == 3 else 0.3)).astype(np.float32)
    x = (r.randn(p["in_channels"], p["height"], p["width"]) * 0.5).astype(np.float32)
    b = (r.randn(out_ch) * 0.3).astype(np.float32) if with_bias else np.zeros(out_ch, np.float32)
    if mode == "negative":
        w, x = -np.ones_like(w), np.full_like(x, 5.0)
    elif mode == "negative_bias":
        w, x, b = np.zeros_like(w), np.ones_like(x), -np.abs(b) - 0.1
    return w, x, b


# pointwise_conv_tpl: x, w [out, in, 1, 1], b (present when apply_relu, the seeded program)
def _pointwise(plan, mode="random"):
    p = plan.params
    w, x, b = _draw(
        plan,
        out_ch=p["out_channels"],
        group_in=p["in_channels"],
        k=1,
        with_bias=p["apply_relu"],
        mode=mode,
    )
    return {"x": x, "w": w, **({"b": b} if p["apply_relu"] else {})}


def legacy_pointwise(plan, rng):
    return _pointwise(plan)


def legacy_pointwise_negative_bias(plan, rng):
    return _pointwise(plan, "negative_bias")


# depthwise_conv_tpl and the stride-2 kernels: x, filters [C, 10] (bias, then the 9 taps;
# the bias zero for the plain twin, which drew none)
def _depthwise(plan, mode="random"):
    p = plan.params
    c = p["in_channels"]
    seeded = p["has_bias"] or p["apply_relu"]
    w, x, b = _draw(plan, out_ch=c, group_in=1, k=3, with_bias=seeded, mode=mode)
    return {"x": x, "filters": np.concatenate([b[:, None], w.reshape(c, 9)], axis=1)}


def legacy_depthwise(plan, rng):
    return _depthwise(plan)


def legacy_depthwise_negative(plan, rng):
    return _depthwise(plan, "negative")


def legacy_depthwise_negative_bias(plan, rng):
    return _depthwise(plan, "negative_bias")
