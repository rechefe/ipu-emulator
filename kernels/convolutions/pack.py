"""The convolutions' pack transforms (E07-16), shared by the family's kernels.

The tiled kernels (``conv1x1``, ``conv3x3_relu``, ``conv3x3_relu_cin1``) read
``ipu_apps``' tiled fixture layout, now laid out one tensor at a time:

* ``tiled_guard``, ``halo_tiled_guard``, ``halo_tiled``: the input,
  ``(channels, height, width)`` cut into tiles of ``128 - 2*halo`` columns,
  one 128-lane row per (channel, padded row, tile); lane ``j`` of tile ``t``
  is column ``t*(128 - 2*halo) + j - halo``, zero outside the image, and
  ``halo`` all-zero rows above and below each channel. ``guard`` adds one
  all-zero channel plane: the plane the pipelined channel loop prefetches
  past the last channel. It belongs to the input's region because the program
  never reads its base.
* ``halo_tiled_output``: the 3x3 kernels' output, one row per (channel, row,
  tile) with lanes 0..125 holding columns ``t*126 .. t*126 + 125``; the other
  two lanes are dropped on unpack.
* ``conv_weight_groups``: ``(out, in, k, k)`` weights, ``ceil(in / g)`` rows
  per output channel with ``g = 128 // k**2`` input channels a row (128 for
  1x1, 14 for 3x3): row ``o*G + gi`` holds ``w[o, gi*g:(gi+1)*g].ravel()``.
* ``lane0_rows``: ``(n,)`` as n rows, value ``i`` in lane 0 of row ``i``
  (the bias rows).

``conv_first_layer`` stores its input and output row-interleaved by channel
(``row_interleaved``: ``(c, h, w)`` stored as ``(h, c, w)`` row-major).

The 1x1 output is the row-major default: ``ceil(width / 128)`` rows per
(channel, row), as ``tiled_guard`` without the guard plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

Array = Any


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


class HaloTiled:
    """``(channels, height, width)`` as halo tiles, optionally with a guard plane."""

    def __init__(self, halo: int, guard: bool) -> None:
        self.halo = halo
        self.guard = guard
        self.columns = LANES - 2 * halo

    def _dims(self, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        if len(shape) != 3:
            raise ValueError(f"halo tiles lay out (channels, height, width), not {shape}")
        c, h, w = shape
        return c, h, w, _ceil(w, self.columns)

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """``(channels + guard) * (height + 2*halo) * tiles`` rows.

        Raises:
            ValueError: Not three dimensions.
        """
        c, h, _, tiles = self._dims(shape)
        return (c + int(self.guard)) * (h + 2 * self.halo) * tiles

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``: tiles, halo rows and the guard plane zero-filled."""
        c, h, w, tiles = self._dims(a.shape)
        packed = np.zeros((c + int(self.guard), h + 2 * self.halo, tiles, LANES), dtype=np.float64)
        for tile in range(tiles):
            source = np.arange(LANES) + tile * self.columns - self.halo
            valid = (source >= 0) & (source < w)
            packed[:c, self.halo : self.halo + h, tile, valid] = a[:, :, source[valid]]
        return packed.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`)."""
        c, h, w, tiles = self._dims(shape)
        t = r.reshape(c + int(self.guard), h + 2 * self.halo, tiles, LANES)
        t = t[:c, self.halo : self.halo + h, :, self.halo : self.halo + self.columns]
        return t.reshape(c, h, -1)[..., :w]


class HaloTiledOutput:
    """``(channels, height, width)`` output of halo-tiled kernels: one row per
    (channel, row, tile), the tile's ``128 - 2*halo`` output columns in lanes
    ``0 .. 127 - 2*halo``; the last ``2*halo`` lanes (what the taps past the
    tile leave there) are dropped on unpack."""

    def __init__(self, halo: int) -> None:
        self.columns = LANES - 2 * halo

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """One row per (channel, row, tile)."""
        c, h, w = shape
        return c * h * _ceil(w, self.columns)

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``, the lanes past the columns zero."""
        c, h, w = a.shape
        tiles = _ceil(w, self.columns)
        wide = np.zeros((c, h, tiles * self.columns), dtype=np.float64)
        wide[..., :w] = a
        rows = np.zeros((c, h, tiles, LANES), dtype=np.float64)
        rows[..., : self.columns] = wide.reshape(c, h, tiles, self.columns)
        return rows.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds; the lanes past the columns are dropped."""
        c, h, w = shape
        tiles = _ceil(w, self.columns)
        return r.reshape(c, h, tiles, LANES)[..., : self.columns].reshape(c, h, -1)[..., :w]


class ConvWeightGroups:
    """``(out, in, k, k)`` weights in groups of ``128 // k**2`` input channels a row."""

    @staticmethod
    def _group(shape: tuple[int, ...]) -> int:
        if len(shape) != 4 or shape[2] != shape[3]:
            raise ValueError(f"conv_weight_groups lays out (out, in, k, k), not {shape}")
        return LANES // (shape[2] * shape[3])

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """``ceil(in / g)`` rows per output channel.

        Raises:
            ValueError: Not ``(out, in, k, k)``.
        """
        return shape[0] * _ceil(shape[1], self._group(shape))

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``, each group's ``g * k * k`` weights then zeros."""
        g = self._group(a.shape)
        o, c = a.shape[:2]
        groups = _ceil(c, g)
        rows = np.zeros((o, groups, LANES), dtype=np.float64)
        for gi in range(groups):
            chunk = a[:, gi * g : (gi + 1) * g].reshape(o, -1)
            rows[:, gi, : chunk.shape[1]] = chunk
        return rows.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`)."""
        g = self._group(shape)
        o, c, k, _ = shape
        rr = r.reshape(o, _ceil(c, g), LANES)
        out = np.zeros(shape, dtype=np.float64)
        for gi in range(_ceil(c, g)):
            n = min(g, c - gi * g)
            out[:, gi * g : gi * g + n] = rr[:, gi, : n * k * k].reshape(o, n, k, k)
        return out


class Lane0Rows:
    """``(n,)`` as n rows, value ``i`` in lane 0 of row ``i``."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """One row per value.

        Raises:
            ValueError: Not one dimension.
        """
        if len(shape) != 1:
            raise ValueError(f"lane0_rows lays out (n,), not {shape}")
        return shape[0]

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` in lane 0 of each row, the other lanes zero."""
        rows = np.zeros((a.shape[0], LANES), dtype=np.float64)
        rows[:, 0] = a
        return rows

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """Lane 0 of each row."""
        return r[: shape[0], 0].copy()


class RowInterleaved:
    """``(channels, height, width)`` stored ``(height, channels, width)``, row-major."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """``ceil(width / 128)`` rows per (row, channel)."""
        c, h, w = shape
        return h * c * _ceil(w, LANES)

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` transposed to ``(height, channels, width)``, each run zero-padded."""
        c, h, w = a.shape
        t = np.zeros((h, c, _ceil(w, LANES) * LANES), dtype=np.float64)
        t[..., :w] = np.transpose(a, (1, 0, 2))
        return t.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`)."""
        c, h, w = shape
        return np.transpose(r.reshape(h, c, -1)[..., :w], (1, 0, 2))


tiled_guard = HaloTiled(halo=0, guard=True)
halo_tiled_guard = HaloTiled(halo=1, guard=True)
halo_tiled = HaloTiled(halo=1, guard=False)
halo_tiled_output = HaloTiledOutput(halo=1)
conv_weight_groups = ConvWeightGroups()
lane0_rows = Lane0Rows()
row_interleaved = RowInterleaved()
