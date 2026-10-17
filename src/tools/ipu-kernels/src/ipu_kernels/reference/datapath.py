"""Datapath mirrors: a contraction rounded exactly as the emulator rounds it (E07-12).

A math reference (``ipu_kernels.reference.ops``) holds a kernel to a tolerance
tier. These hold it bit for bit (``EXACT``), by repeating the datapath's own
rounding in NumPy, so a family test can check both: a bug the mirror shares
with the emulator is still caught by the math reference, and a rounding
change the tier would hide is caught here (E06-09).

Both compute ``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]``, lane ``i``,
contracted over ``s``:

* :func:`agg_fold`, ``MULT.RC.VV`` + ``AGG.SUM[.FIRST]``: float32 lane
  products, a float64 left fold per 128-lane chunk rounded once to float32 on
  the R_ACC write, and each later chunk added in float32 to the rounded
  partial;
* :func:`acc_fold`, ``MULT.RC.VE`` + ``ACC.ADD[.FIRST]``: a float32 running
  sum per lane, rounded at every step.

Moved from ``ipu_apps.kernels.attention.cases``.
"""

from __future__ import annotations

import numpy as np

from ipu_isa.geometry import LANES

__all__ = ["acc_fold", "agg_fold"]


def agg_fold(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]`` through the AGG datapath.

    ``AGG.SUM.FIRST`` left-folds the active MULT_RES lanes from a float64 and
    rounds the total once to float32 on the R_ACC write; the lane products are
    float32. A second 128-element chunk is a second float64 fold, added by
    ``AGG.SUM`` in float32 to the already-rounded partial. ``np.cumsum`` over
    float64 is a strict left fold, so this is the scalar loop, vectorised.

    Args:
        P: ``[b, i, s]``.
        V: ``[b, t, s]``.

    Returns:
        ``O``, ``[b, t, i]`` float32.
    """
    prod = (P[:, None, :, :].astype(np.float32) * V[:, :, None, :].astype(np.float32)).astype(
        np.float32
    )  # [b, t, i, s]
    out = None
    for lo in range(0, prod.shape[-1], LANES):
        chunk = np.cumsum(prod[..., lo : lo + LANES].astype(np.float64), axis=-1)[..., -1]
        chunk = chunk.astype(np.float32)
        out = (
            chunk
            if out is None
            else (chunk.astype(np.float64) + out.astype(np.float64)).astype(np.float32)
        )
    assert out is not None, "an empty contraction"
    return out


def acc_fold(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]`` through the ACC datapath.

    Each step forms float32 lane products and ``ACC.ADD`` writes the running
    sum back as float32; ``ACC.ADD.FIRST`` seeds it at ``s = 0``. One float32
    left fold over ``s``, rounded at every step.

    Args:
        P: ``[b, i, s]``.
        V: ``[b, t, s]``.

    Returns:
        ``O``, ``[b, t, i]`` float32.
    """
    P = P.astype(np.float32)
    V = V.astype(np.float32)
    acc = None
    for s in range(P.shape[-1]):
        prod = (P[:, None, :, s] * V[:, :, None, s]).astype(np.float32)  # [b, t, i]
        acc = prod if acc is None else (acc + prod).astype(np.float32)
    assert acc is not None, "an empty contraction"
    return acc
