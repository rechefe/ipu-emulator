"""The reshape family's pack transforms (E07-04, E07-05), loaded by path.

``tile_padded`` lays depth_to_space's output ``[h * r, w * r]`` out as the
program stores it: each output row takes ``ceil(w / 128) * r`` whole XMEM
rows, the input's 128-lane tiles shuffled, so its first ``w * r`` lanes are
the row and the rest the shuffled zero padding, dropped on unpack.

``stripes`` is a ``[c, h, w]`` tensor in ``n = ceil(h * w / 128)`` stripes of
``h / n`` spatial rows: row ``s * c + k`` holds channel ``k``'s stripe ``s``,
row-major, zero-padded (unfold's input, fold's output; for an 8 x 8 tensor one
stripe of 64 lanes).

``paired_rows`` is unfold_8x8x240's input ``[c, 8, 8]``: one row per channel,
the spatial rows in the order (0, 2, 1, 3, 4, 6, 5, 7), so ACC.STRIDE's
even/odd split of the view rows is the even/odd split of the spatial rows.
Lanes 64..127 hold the case's ``pad`` parameter (0 unless a case sets it).

``phase_rows`` is the four stride-2 phases ``[4, c, tokens]`` (unfold's
output, fold's input): row-major, ``ceil(tokens / 128)`` rows per channel,
the lanes past the tokens NaN when packed (a fold that read them would
fail), dropped on unpack (unfold leaves stale lanes there).
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


def _stripe_count(shape: tuple[int, ...]) -> int:
    _, h, w = shape
    n = math.ceil(h * w / LANES)
    if h % n or (h // n) * w > LANES:
        raise ValueError(f"stripes: {h} x {w} does not split into stripes of whole rows")
    return n


class Stripes:
    """``[c, h, w]`` as ``n`` stripes of ``h / n`` spatial rows, stripe-major."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        return _stripe_count(shape) * shape[0]

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        c, h, w = a.shape
        n = _stripe_count(a.shape)
        out = np.zeros((n, c, LANES), dtype=np.float64)
        out[..., : (h // n) * w] = a.reshape(c, n, (h // n) * w).transpose(1, 0, 2)
        return out.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        c, h, w = shape
        n = _stripe_count(shape)
        stripes = r.reshape(n, c, LANES)[..., : (h // n) * w]
        return stripes.transpose(1, 0, 2).reshape(c, h, w)


#: unfold_8x8x240's row order: slot k of a row holds spatial row _PAIRED[k].
_PAIRED = (0, 2, 1, 3, 4, 6, 5, 7)


class PairedRows:
    """``[c, 8, 8]``: one row per channel, spatial rows in ``_PAIRED`` order."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        if tuple(shape[1:]) != (8, 8):
            raise ValueError(f"paired_rows lays out 8 x 8 channels, not {shape[1:]}")
        return shape[0]

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        out = np.full((a.shape[0], LANES), float(plan.params.get("pad", 0.0)))
        for slot, row in enumerate(_PAIRED):
            out[:, slot * 8 : (slot + 1) * 8] = a[:, row, :]
        return out

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        out = np.empty(shape, dtype=np.float64)
        for slot, row in enumerate(_PAIRED):
            out[:, row, :] = r[:, slot * 8 : (slot + 1) * 8]
        return out


class PhaseRows:
    """``[..., tokens]`` row-major, the lanes past the tokens NaN when packed."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        return math.prod(shape[:-1]) * math.ceil(shape[-1] / LANES)

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        width = math.ceil(a.shape[-1] / LANES) * LANES
        out = np.full((*a.shape[:-1], width), np.nan)
        out[..., : a.shape[-1]] = a
        return out.reshape(-1, LANES)

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        width = math.ceil(shape[-1] / LANES) * LANES
        return r.reshape(*shape[:-1], width)[..., : shape[-1]]


stripes = Stripes()
paired_rows = PairedRows()
phase_rows = PhaseRows()
