"""The pooling family's pack transforms (E07-14), loaded by path.

Both read the case's ``padding`` (P): the window kernels' halo, 0 for the
stride-2 kernels.

``halo_tiled`` lays an input ``[C, H, W]`` out as the kernels read it:
channel-major, then padded-row-major, P rows of ``-FLT_MAX`` above and below
each plane, and each padded row cut into ``ceil(W / TC)`` 128-lane tiles of
``TC = 128 - 2P`` columns. Element ``e`` of tile ``t`` is input column
``t * TC + e - P``; columns outside the image hold ``-FLT_MAX``, which no
maximum keeps. With P = 0 it is one XMEM row per 128 columns, ``-FLT_MAX``
past ``W``.

``tiled`` reads an output ``[C, H, W]`` back: ``ceil(W / TC)`` rows per output
row, the first ``TC`` lanes of tile ``t`` output columns ``t * TC ..``; the
other lanes, and the columns past ``W``, are dropped.
"""

import math

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

#: The border value: the identity of a maximum.
FLOOR = float(np.finfo(np.float32).min)


def _columns(plan: CasePlan) -> int:
    """Output columns per tile: ``128 - 2 * padding``."""
    return LANES - 2 * plan.params["padding"]


class HaloTiled:
    """``[C, H, W]`` in halo tiles with ``padding`` border rows, ``-FLT_MAX`` outside."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        c, h, w = shape
        halo = plan.params["padding"]
        return c * (h + 2 * halo) * math.ceil(w / _columns(plan))

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        c, h, w = a.shape
        halo, columns = plan.params["padding"], _columns(plan)
        tiles = math.ceil(w / columns)
        packed = np.full((c, h + 2 * halo, tiles, LANES), FLOOR, dtype=np.float64)
        for tile in range(tiles):
            source = np.arange(LANES) + tile * columns - halo
            valid = (source >= 0) & (source < w)
            packed[:, halo : halo + h, tile, valid] = a[:, :, source[valid]]
        return packed.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        c, h, w = shape
        halo, columns = plan.params["padding"], _columns(plan)
        tiles = math.ceil(w / columns)
        packed = r.reshape(c, h + 2 * halo, tiles, LANES)[:, halo : halo + h]
        source = np.arange(columns)
        return packed[..., halo + source].reshape(c, h, -1)[..., :w]


class Tiled:
    """``[C, H, W]``, ``ceil(W / TC)`` rows per row, the first ``TC`` lanes valid."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        c, h, w = shape
        return c * h * math.ceil(w / _columns(plan))

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        c, h, w = a.shape
        columns = _columns(plan)
        tiles = math.ceil(w / columns)
        padded = np.zeros((c, h, tiles * columns), dtype=np.float64)
        padded[..., :w] = a
        rows = np.zeros((c, h, tiles, LANES), dtype=np.float64)
        rows[..., :columns] = padded.reshape(c, h, tiles, columns)
        return rows.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        c, h, w = shape
        tiles = math.ceil(w / _columns(plan))
        return r.reshape(c, h, tiles, LANES)[..., : _columns(plan)].reshape(c, h, -1)[..., :w]


halo_tiled = HaloTiled()
tiled = Tiled()
