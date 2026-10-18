"""Case generators shared by the projections' templates (E07-02).

``legacy_uniform`` keeps the old cases' data (``cases.py``) in FP32,
so the FP32 outputs compare bit for bit with the old harness's (E02 digests).
In every other format it makes nothing, and the harness's default generator
fills the inputs from their ``range`` (E07-03): the legacy data, uniform in
[-1, 1), would round to {-1, 0, 1} in INT8. ``weight_rows`` is the
weight-pointer check that the old harness's slow ``test.py`` test ran.
``window_mask`` is the packed linear's R_MASK row (E07-23).
"""

import numpy as np

from ipu_isa.formats import FORMATS, ElementFormat
from ipu_kernels.pack import r_mask

#: The old cases' seed.
SEED = 0xC0FFEE

#: weight_rows: output channel j's weights are (j + 1) * SCALE.
SCALE = 0.01


def legacy_uniform(plan, rng):
    """In FP32, b per stream, then a, uniform in [-1, 1) from ``RandomState(SEED)``.

    In another format, nothing: the default generator makes b and a.
    """
    if plan.format is not ElementFormat.FP32:
        return {}
    streams, n_tg, k, n_tok = plan.shapes["b"]
    n_out = plan.shapes["a"][0]
    state = np.random.RandomState(SEED)
    b = np.stack(
        [state.uniform(-1.0, 1.0, size=(n_tg, k, n_tok)).astype(np.float32) for _ in range(streams)]
    )
    a = state.uniform(-1.0, 1.0, size=(n_out, k)).astype(np.float32)
    return {"b": b, "a": a}


def weight_rows(plan, rng):
    """b all ones and a[j, :] = (j + 1) * SCALE, so y[p, tg, j, t] = act((j + 1) * SCALE * k).

    A kernel that resets its per-channel weight-row pointer in the wrong place
    reads channel 0's row for every j: every channel then reads back as
    channel 0's value, which the comparison names at its first wrong index.
    The random-weight cases catch that too; this case makes it the one
    possible signature.
    """
    n_out, k = plan.shapes["a"]
    a = np.repeat(((np.arange(n_out) + 1) * SCALE)[:, None], k, axis=1).astype(np.float32)
    return {"b": np.ones(plan.shapes["b"], dtype=np.float32), "a": a}


def window_mask(plan):
    """packed_output_linear's R_MASK row: each output channel's window in its slot."""
    return r_mask.window_mask(plan.derived["tokens"], FORMATS[plan.format].row_bytes)
