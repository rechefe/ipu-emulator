"""fully_connected's reference: the shared ``ops.linear`` (E07-22).

``ops.linear(a, b)`` is ``a @ b`` with the weights ``a`` ``[n_out, k]`` and
the data ``b`` channel-major ``[k, n_tok]``. ``x`` holds one row per sample,
so the samples are its tokens: ``y = linear(w, x.T).T``. The harness rounds
``y`` once to the accumulator format (INT32 or FP32, saturating), so that is
the golden.
"""

from ipu_kernels.reference.ops import linear


def reference(x, w, **_params):
    """``y[s, :] = w @ x[s, :]``."""
    return {"y": linear(w, x.T)["y"].T}
