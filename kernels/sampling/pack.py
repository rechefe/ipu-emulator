"""The sampling family's pack transform (E07-15).

``hwc_halo`` lays the descriptors ``[H, W, 256]`` out as both kernels read
them: a zero halo cell on every side, ``(H + 2, W + 2, 256)`` in raster
order, two 128-lane rows per cell.
"""

from __future__ import annotations

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

__all__ = ["hwc_halo"]


class HwcHalo:
    """``[H, W, C]`` with one zero cell of halo around it, ``C / 128`` rows per cell."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        h, w, c = shape
        return (h + 2) * (w + 2) * c // LANES

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        h, w, c = a.shape
        padded = np.zeros((h + 2, w + 2, c), dtype=np.float64)
        padded[1:-1, 1:-1] = a
        return padded.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        h, w, c = shape
        return r.reshape(h + 2, w + 2, c)[1:-1, 1:-1]


hwc_halo = HwcHalo()
