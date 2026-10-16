"""The shared transforms, extracted from today's hand-written loaders (E06-05).

Each reshapes one declared tensor into rows of 128 elements and back; the
harness's codec encodes the rows. A tensor's shape is all a transform sees.
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


def _check_width(name: str, width: int) -> None:
    if not 0 < width <= LANES:
        raise ValueError(
            f"{name}: the last dimension ({width}) must be 1..{LANES}, one row's lanes"
        )


class ChannelMajor:
    """``(..., groups, channels, width)`` to one row per ``(channel, group)``:
    the groups of a channel side by side, ``width`` lanes valid and the rest
    zero. The projections' and ``matmul_*_x128`` inputs (``_load_data``: file
    ``(stream, tg, k, tok)``, rows ``(stream, k, tg)``)."""

    def rows(self, shape: tuple[int, ...]) -> int:
        return math.prod(shape[:-1])

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        _check_width("channel_major", a.shape[-1])
        swapped = np.swapaxes(a, -3, -2)
        rows = np.zeros((*swapped.shape[:-1], LANES), dtype=np.float64)
        rows[..., : a.shape[-1]] = swapped
        return rows.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        *lead, groups, channels, width = shape
        swapped = r.reshape(*lead, channels, groups, LANES)[..., :width]
        return np.swapaxes(swapped, -3, -2)


class ChunkInterleaved:
    """``(channels, rows, cols)`` to 128-element chunks: ``128 // cols`` spatial
    rows share a chunk, and a chunk holds every channel in turn
    (``pack_input_chunked`` in the convolutions: element
    ``(chunk * channels + ch) * 128 + local_row * cols + col``)."""

    def rows(self, shape: tuple[int, ...]) -> int:
        channels, height, cols = shape
        return -(-height // (LANES // cols)) * channels

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        channels, height, cols = a.shape
        if cols <= 0 or LANES % cols:
            raise ValueError(
                f"chunk_interleaved: cols ({cols}) must divide {LANES}, so a spatial row "
                f"tiles one chunk without straddling its edge"
            )
        per_chunk = LANES // cols
        chunks = -(-height // per_chunk)
        tall = np.zeros((channels, chunks * per_chunk, cols), dtype=np.float64)
        tall[:, :height] = a
        flat = tall.reshape(channels, chunks, per_chunk * cols).transpose(1, 0, 2)
        rows = np.zeros((chunks, channels, LANES), dtype=np.float64)
        rows[..., : per_chunk * cols] = flat
        return rows.reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        channels, height, cols = shape
        per_chunk = LANES // cols
        chunks = -(-height // per_chunk)
        flat = r.reshape(chunks, channels, LANES)[..., : per_chunk * cols].transpose(1, 0, 2)
        return flat.reshape(channels, chunks * per_chunk, cols)[:, :height]


#: ``(n_out, k)`` weights, one output channel's ``k`` values per ``ceil(k / 128)``
#: rows, zero-padded: the row-major default, named for the projections' and
#: ``matmul_*_x128`` weights (``_load_weights``).
output_major_weights = RowMajor()
channel_major = ChannelMajor()
chunk_interleaved = ChunkInterleaved()
