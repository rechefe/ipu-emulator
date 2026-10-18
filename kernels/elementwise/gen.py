"""The elementwise family's case data (E07-06), shared by the residual_add_rows entries.

Each entry's default case draws ``RandomState(seed)`` with the entry's seed,
``a`` then ``b``, uniform in [-bound, bound) in FP32 over the logical
``(channels, tokens)`` tensor, laid out ``per_row`` channels a row. The bound
is 1 for the unpacked entries and 3 for the packed ones (E07-23).
"""

import numpy as np

#: Each entry's seed (the template's own default case takes 0).
SEEDS = {
    "residual_add_16x240": 0x5ADD,
    "residual_add_64x192": 0x64192,
    "residual_add_256x144": 0xADD,
    "packed_residual_add_192x64": 7,
    "packed_residual_add_240x16": 7,
}

#: Each entry's bound, where it is not 1.
BOUNDS = {
    "packed_residual_add_192x64": 3.0,
    "packed_residual_add_240x16": 3.0,
}


def seeded_uniform(plan, rng):
    p = plan.params
    channels, tokens, per_row = p["channels"], p["tokens"], p["per_row"]
    old = np.random.RandomState(SEEDS.get(plan.manifest.name, 0))
    bound = BOUNDS.get(plan.manifest.name, 1.0)
    out = {}
    for name in ("a", "b"):
        x = old.uniform(-bound, bound, size=(channels, tokens)).astype(np.float32)
        out[name] = x.reshape(channels // per_row, per_row * tokens).astype(np.float64)
    return out
