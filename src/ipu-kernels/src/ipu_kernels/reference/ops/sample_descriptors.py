"""Dense 8x bilinear descriptor sampling, HWC (E07-15)."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["sample_descriptors"]


def _grid(n: int, start: int, count: int, stock: bool) -> np.ndarray:
    """The coarse-grid coordinate of each of ``count`` fine positions from
    ``start``, in FP32 as the kernels' coefficient tables were built.

    Shared: the fine pixel centre, ``(p - 3.5) / 8``. Stock: SuperPoint's
    ``sample_descriptors``, the normalized grid ``((p - 3.5) / (8n - 4.5)) *
    2 - 1`` mapped back with ``align_corners=True``.
    """
    p = np.arange(start, start + count, dtype=np.float32)
    if not stock:
        return (p - np.float32(3.5)) / np.float32(8)
    grid = ((p - np.float32(3.5)) / np.float32(8 * n - 4.5)) * np.float32(2) - np.float32(1)
    return ((grid + np.float32(1)) / np.float32(2)) * np.float32(n - 1)


def sample_descriptors(
    x: torch.Tensor,
    *,
    stock: bool,
    cell_row_start: int,
    cell_row_count: int,
    **_params: object,
) -> dict[str, torch.Tensor]:
    """Sample ``x`` ``[H, W, C]`` at 8 fine positions per coarse cell along
    each axis, for the coarse rows ``cell_row_start`` onward
    (``cell_row_count`` of them, 0: to the last): ``y`` is
    ``[8 * rows, 8 * W, C]``.

    Bilinear by explicit indexing (not ``grid_sample``): each fine position
    weights its four coarse neighbours by ``1 - |offset|``; a neighbour
    outside the map contributes zero (``padding_mode="zeros"``). ``stock``
    selects SuperPoint's ``align_corners=True`` coordinates, else the shared
    phase-centred ones (:func:`_grid`). No L2 normalization.
    """
    h, w, _ = x.shape
    rows = cell_row_count or h - cell_row_start
    ys = _grid(h, 8 * cell_row_start, 8 * rows, stock).astype(np.float64)
    xs = _grid(w, 0, 8 * w, stock).astype(np.float64)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - y0, xs - x0
    out = torch.zeros((len(ys), len(xs), x.shape[2]), dtype=x.dtype)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            iy, ix = y0 + dy, x0 + dx
            valid = ((iy >= 0) & (iy < h))[:, None] & ((ix >= 0) & (ix < w))[None, :]
            weight = torch.from_numpy(wy[:, None] * wx[None, :] * valid)
            gathered = x[torch.from_numpy(np.clip(iy, 0, h - 1))][
                :, torch.from_numpy(np.clip(ix, 0, w - 1))
            ]
            out += gathered * weight[..., None].to(x.dtype)
    return {"y": out}
