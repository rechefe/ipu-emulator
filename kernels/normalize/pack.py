"""The normalize family's pack transforms (E07-23)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

Array = npt.NDArray[Any]


class ChannelTile:
    """A per-channel ``[channels]`` vector in the packed layout: channel ``c``'s
    value in every lane of its window, row ``c // pack``, lanes
    ``(c % pack) * tokens`` on (``pack = 128 // tokens``): packed_layernorm's
    gamma and beta."""

    @staticmethod
    def _tokens(shape: tuple[int, ...], plan: CasePlan) -> int:
        tokens = int(plan.params["tokens"])
        if len(shape) != 1 or LANES % tokens or shape[0] % (LANES // tokens):
            raise ValueError(
                f"channel_tile lays out [channels] with 128 / tokens ({tokens}) "
                f"channels a row, not {shape}"
            )
        return tokens

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        tokens = self._tokens(shape, plan)
        return shape[0] * tokens // LANES

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        tokens = self._tokens(a.shape, plan)
        return np.repeat(np.asarray(a, dtype=np.float64), tokens).reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        tokens = self._tokens(shape, plan)
        return r.reshape(-1)[::tokens][: shape[0]]


channel_tile = ChannelTile()
