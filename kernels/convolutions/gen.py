"""The stride-2 depthwise and wide384 cases' ipu_apps data (E07-19), for the proof only.

generator = "../gen.py:<fn>": convolutions/cases.py's _data, its seed (the
old case's, the stride-2 kernels' ``seed`` parameter; 0 for wide384) and draw
order (weights randn * 0.2, then the input randn * 0.5; none of these kernels
had a bias), so the outputs compare bit for bit with the old harness's. The
next commit moves the cases to the harness's seeded generator and deletes
this file.
"""

import numpy as np


def _draw(plan, *, out_ch, group_in):
    p = plan.params
    r = np.random.RandomState(p.get("seed", 0))
    w = (r.randn(out_ch, group_in, 3, 3) * 0.2).astype(np.float32)
    x = (r.randn(p["in_channels"], p["height"], p["width"]) * 0.5).astype(np.float32)
    return w, x


# The stride-2 kernels: x, filters [C, 10] (a zero bias column, then the 9 taps)
def legacy_depthwise(plan, rng):
    c = plan.params["in_channels"]
    w, x = _draw(plan, out_ch=c, group_in=1)
    b = np.zeros((c, 1), np.float32)
    return {"x": x, "filters": np.concatenate([b, w.reshape(c, 9)], axis=1)}


# conv_universal_wide384: x, w [out, in, 3, 3]
def legacy_wide(plan, rng):
    p = plan.params
    w, x = _draw(plan, out_ch=p["out_channels"], group_in=p["in_channels"])
    return {"x": x, "w": w}
