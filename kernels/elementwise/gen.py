"""The elementwise family's case data (E07-06), shared by the residual_add_rows entries.

Each entry's default case keeps the old ``ipu_apps`` data: ``RandomState(seed)``
with the entry's old seed, ``a`` then ``b``, uniform in [-1, 1) in FP32 over the
logical ``(channels, tokens)`` tensor, laid out ``per_row`` channels a row.
"""

import numpy as np

#: Each kernel's old seed (the template's own default case takes 0).
SEEDS = {
    "residual_add_16x240": 0x5ADD,
    "residual_add_64x192": 0x64192,
    "residual_add_256x144": 0xADD,
}


def legacy_uniform(plan, rng):
    p = plan.params
    channels, tokens, per_row = p["channels"], p["tokens"], p["per_row"]
    old = np.random.RandomState(SEEDS.get(plan.manifest.name, 0))
    out = {}
    for name in ("a", "b"):
        x = old.uniform(-1.0, 1.0, size=(channels, tokens)).astype(np.float32)
        out[name] = x.reshape(channels // per_row, per_row * tokens).astype(np.float64)
    return out
