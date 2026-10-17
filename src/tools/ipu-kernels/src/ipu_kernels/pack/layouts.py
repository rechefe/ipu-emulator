"""The shared transforms, extracted from today's hand-written loaders (E06-05).

Each reshapes one declared tensor into rows of 128 elements and back; the
harness's codec encodes the rows. A tensor's shape is all a transform sees. A
shape a transform cannot lay out is a ``ValueError``, which packing reports as
a configuration error located at the tensor's ``pack`` key.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_harness.packing import RowMajor
from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

Array = npt.NDArray[Any]


class ChannelMajor:
    """``(..., groups, channels, width)`` to one row per ``(channel, group)``:
    the groups of a channel side by side, ``width`` lanes valid and the rest
    zero. The projections' and ``matmul_*_x128`` inputs (``_load_data``: file
    ``(stream, tg, k, tok)``, rows ``(stream, k, tg)``)."""

    @staticmethod
    def _check(shape: tuple[int, ...]) -> None:
        if len(shape) < 3:
            raise ValueError(f"channel_major lays out (..., groups, channels, width), not {shape}")
        if not 0 < shape[-1] <= LANES:
            raise ValueError(f"channel_major: the width ({shape[-1]}) must fit one row, 1..{LANES}")

    def rows(self, shape: tuple[int, ...], plan: CasePlan | None = None) -> int:
        """One row per index of every dimension but the width.

        Raises:
            ValueError: Fewer than three dimensions, or a width over 128.
        """
        self._check(shape)
        return math.prod(shape[:-1])

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``: channel-major, each row zero-padded.

        Raises:
            ValueError: As :meth:`rows`.
        """
        self._check(a.shape)
        swapped = np.swapaxes(a, -3, -2)
        rows = np.zeros((*swapped.shape[:-1], LANES), dtype=np.float64)
        rows[..., : a.shape[-1]] = swapped
        return rows.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`).

        Raises:
            ValueError: As :meth:`rows`.
        """
        self._check(shape)
        *lead, groups, channels, width = shape
        swapped = r.reshape(*lead, channels, groups, LANES)[..., :width]
        return np.swapaxes(swapped, -3, -2)


class ChunkInterleaved:
    """``(channels, rows, cols)`` to 128-element chunks: ``128 // cols`` spatial
    rows share a chunk, and a chunk holds every channel in turn
    (``pack_input_chunked`` in the convolutions: element
    ``(chunk * channels + ch) * 128 + local_row * cols + col``)."""

    @staticmethod
    def _per_chunk(shape: tuple[int, ...]) -> int:
        if len(shape) != 3:
            raise ValueError(f"chunk_interleaved lays out (channels, rows, cols), not {shape}")
        cols = shape[-1]
        if cols <= 0 or LANES % cols:
            raise ValueError(
                f"chunk_interleaved: cols ({cols}) must divide {LANES}, so a spatial row "
                f"tiles one chunk without straddling its edge"
            )
        return LANES // cols

    def rows(self, shape: tuple[int, ...], plan: CasePlan | None = None) -> int:
        """``ceil(rows / (128 // cols))`` chunks of every channel.

        Raises:
            ValueError: Not three dimensions, or cols that do not divide 128.
        """
        channels, height, _ = shape
        return -(-height // self._per_chunk(shape)) * channels

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        """``a`` as ``(rows, 128)``, chunk-interleaved, the last chunk zero-padded.

        Raises:
            ValueError: As :meth:`rows`.
        """
        per_chunk = self._per_chunk(a.shape)
        channels, height, cols = a.shape
        chunks = -(-height // per_chunk)
        tall = np.zeros((channels, chunks * per_chunk, cols), dtype=np.float64)
        tall[:, :height] = a
        # cols divides 128, so a chunk of per_chunk rows fills a row exactly.
        return tall.reshape(channels, chunks, LANES).transpose(1, 0, 2).reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        """The ``shape`` tensor ``r`` holds (inverse of :meth:`to_rows`).

        Raises:
            ValueError: As :meth:`rows`.
        """
        per_chunk = self._per_chunk(shape)
        channels, height, cols = shape
        chunks = -(-height // per_chunk)
        tall = r.reshape(chunks, channels, LANES).transpose(1, 0, 2)
        return tall.reshape(channels, chunks * per_chunk, cols)[:, :height]


#: ``(n_out, k)`` weights, one output channel's ``k`` values per ``ceil(k / 128)``
#: rows, zero-padded: the row-major default, named for the projections' and
#: ``matmul_*_x128`` weights (``_load_weights``).
output_major_weights = RowMajor()
channel_major = ChannelMajor()
chunk_interleaved = ChunkInterleaved()
