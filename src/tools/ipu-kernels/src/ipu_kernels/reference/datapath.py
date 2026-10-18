"""Datapath mirrors: a contraction rounded exactly as the emulator rounds it (E07-12).

A math reference (``ipu_kernels.reference.ops``) holds a kernel to a tolerance
tier. These hold it bit for bit (``EXACT``), by repeating the datapath's own
rounding in NumPy, so a family test can check both: a bug the mirror shares
with the emulator is still caught by the math reference, and a rounding
change the tier would hide is caught here (E06-09).

Both compute ``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]``, lane ``i``,
contracted over ``s``:

* :func:`agg_fold`, ``MULT.RC.VV`` + ``AGG.SUM[.FIRST]``: float32 lane
  products, a float64 left fold per 128-lane chunk, and the R_ACC write
  rounding once to float32: a later chunk's fold, unrounded, is added in
  float64 to the partial R_ACC holds, then rounded;
* :func:`acc_fold`, ``MULT.RC.VE`` + ``ACC.ADD[.FIRST]``: a float32 running
  sum per lane, rounded at every step.

:func:`agg_fold` adds a later chunk's fold to the partial unrounded, as
``AGG.SUM`` does (E07-13); rounding it first is a bug an ``rtol = 1e-6`` check
hides.
"""

from __future__ import annotations

import numpy as np

from ipu_isa.geometry import LANES

__all__ = ["acc_fold", "acc_mirror", "agg_fold", "agg_mirror"]


def agg_fold(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``O[b, t, i] = sum_s P[b, i, s] * V[b, t, s]`` through the AGG datapath.

    ``AGG.SUM.FIRST`` left-folds the active MULT_RES lanes from a float64 and
    rounds the total once to float32 on the R_ACC write; the lane products are
    float32. A second 128-element chunk is a second float64 fold, which
    ``AGG.SUM`` adds, unrounded, to the float32 partial in R_ACC, rounding the
    sum once on the write. ``np.cumsum`` over float64 is a strict left fold, so
    this is the scalar loop, vectorised.

    Args:
        P: ``[b, i, s]``.
        V: ``[b, t, s]``.

    Returns:
        ``O``, ``[b, t, i]`` float32.

    Raises:
        ValueError: ``s`` is empty.
    """
    prod = (P[:, None, :, :].astype(np.float32) * V[:, :, None, :].astype(np.float32)).astype(
        np.float32
    )  # [b, t, i, s]
    out = None
    for lo in range(0, prod.shape[-1], LANES):
        # + 0.0: the emulator's fold starts from 0, so an all -0.0 chunk sums to +0.0.
        chunk = np.cumsum(prod[..., lo : lo + LANES].astype(np.float64), axis=-1)[..., -1] + 0.0
        total = chunk if out is None else chunk + out.astype(np.float64)
        out = total.astype(np.float32)
    if out is None:
        raise ValueError("agg_fold: an empty contraction (no s)")
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

    Raises:
        ValueError: ``s`` is empty.
    """
    P = P.astype(np.float32)
    V = V.astype(np.float32)
    acc = None
    for s in range(P.shape[-1]):
        prod = (P[:, None, :, s] * V[:, :, None, s]).astype(np.float32)  # [b, t, i]
        acc = prod if acc is None else (acc + prod).astype(np.float32)
    if acc is None:
        raise ValueError("acc_fold: an empty contraction (no s)")
    return acc


def acc_mirror(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b`` as the ACC datapath rounds it, batched over leading axes.

    ``a`` is ``[..., rows, k]`` and ``b`` ``[..., k, lanes]``: lane ``l`` of row
    ``r`` folds ``b[c, l] * a[r, c]`` over ``c``, i.e. :func:`acc_fold` with
    ``P = b`` transposed and ``V = a`` (the score kernels, ``attn_v_bcast_*``).
    """
    a3, b3 = a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])
    return acc_fold(np.swapaxes(b3, -1, -2), a3).reshape(*a.shape[:-1], b.shape[-1])


def agg_mirror(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a @ b^T`` as the AGG datapath rounds it, batched over leading axes.

    ``b`` is P ``[..., i, s]`` and ``a`` is V ``[..., t, s]``:
    :func:`agg_fold` ``(P = b, V = a)`` (``attn_v_*``).
    """
    a3, b3 = a.reshape(-1, *a.shape[-2:]), b.reshape(-1, *b.shape[-2:])
    return agg_fold(b3, a3).reshape(*a.shape[:-1], b.shape[-2])
