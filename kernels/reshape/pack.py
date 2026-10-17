"""The reshape family's pack transforms (E07-04, E07-05), loaded by path.

``tile_padded`` lays depth_to_space's output ``[h * r, w * r]`` out as the
program stores it: each output row takes ``ceil(w / 128) * r`` whole XMEM
rows, the input's 128-lane tiles shuffled, so its first ``w * r`` lanes are
the row and the rest the shuffled zero padding, dropped on unpack.
"""

import math

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES


class TilePadded:
    """``[h * r, w * r]``, each row padded to ``ceil(w / 128) * r * 128`` lanes."""

    def _width(self, plan: CasePlan) -> int:
        return math.ceil(plan.params["w"] / LANES) * plan.params["upscale_factor"] * LANES

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        return shape[0] * self._width(plan) // LANES

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        padded = np.zeros((a.shape[0], self._width(plan)), dtype=np.float64)
        padded[:, : a.shape[1]] = a
        return padded.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        return r.reshape(shape[0], self._width(plan))[:, : shape[1]]


tile_padded = TilePadded()
