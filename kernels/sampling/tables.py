"""The sampling kernels' coefficient tables (E07-15): scratch regions with
``bytes`` content, built from the geometry alone (the descriptors never enter).

``shared`` and ``stock`` are ``sample_descriptors``' tables K: per fine
phase (a, b) the weight of each of the nine coarse neighbours (dy, dx), nine
rows per table, phases ``8 * a + b`` in lanes 0..63. Shared mode has one table
for every cell; stock mode one per cell, ``K[i, j]`` from SuperPoint's
``align_corners=True`` grid. ``ky`` and ``kx`` are
``sample_descriptors_separable``'s per-axis tables: one row per coarse row
(column), ``[tap, phase]`` in lanes 0..23. The arithmetic is the old
``sampling/cases.py``'s, in FP32.
"""

import numpy as np

LANES = 128
OFFSETS = np.arange(-1, 2, dtype=np.float32)


def _axis(n):
    """Stock coefficients along one axis of ``n`` cells: ``[n, 3 taps, 8 phases]``."""
    p = np.arange(8 * n, dtype=np.float32) - np.float32(3.5)
    p /= np.float32(8 * n - 4.5)
    p = p * np.float32(2) - np.float32(1)
    p = (p + np.float32(1)) / np.float32(2) * np.float32(n - 1)
    delta = p.reshape(n, 8)[:, None, :] - np.arange(n, dtype=np.float32)[:, None, None]
    return np.maximum(np.float32(0), np.float32(1) - np.abs(delta - OFFSETS[None, :, None]))


def _rows(values, width, rows):
    """``values`` (one table row each) in the first ``width`` lanes of ``rows`` rows."""
    image = np.zeros((rows, LANES), dtype="<f4")
    image[:, :width] = np.asarray(values, dtype="<f4").reshape(rows, width)
    return image.tobytes()


def shared(plan):
    """``K[dy, dx, a, b]``: the phase-centred bilinear weights, one table."""
    t = (np.arange(8, dtype=np.float32) - np.float32(3.5)) / np.float32(8)
    k = np.maximum(np.float32(0), np.float32(1) - np.abs(t[None, :] - OFFSETS[:, None]))
    return _rows(k[:, None, :, None] * k[None, :, None, :], 64, 9)


def stock(plan):
    """``K[i, j, dy, dx, a, b]``: one table per coarse cell (stock coordinates)."""
    h, w = plan.params["height"], plan.params["width"]
    ky, kx = _axis(h), _axis(w)
    return _rows(ky[:, None, :, None, :, None] * kx[None, :, None, :, None, :], 64, h * w * 9)


def ky(plan):
    """``Ky[i, dy, a]``: one row per coarse row."""
    h = plan.params["height"]
    return _rows(_axis(h), 24, h)


def kx(plan):
    """``Kx[j, dx, b]``: one row per coarse column."""
    w = plan.params["width"]
    return _rows(_axis(w), 24, w)
