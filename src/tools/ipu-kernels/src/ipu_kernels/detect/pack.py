"""The detect family's pack transforms (E07-15).

``padded_scores`` lays ``score_threshold``'s scores ``[count]`` out one row
per 128, and fills the lanes past ``count`` with ``tau - 800 / T`` (FP32
arithmetic, as the program assumes): ``T * (pad - tau) = -800``, so the
padding's sigmoid underflows to 0 and its ReLU is 0, and it adds nothing to
either output.

``gate_and_count`` reads ``score_threshold``'s output ``[count + 1]`` back:
the selected scores, one row per 128 (``count`` values), then the soft count
in lane 0 of the row after them. The other lanes are dropped; the kernel's
own test holds the selected rows' padding lanes to 0.
"""

from __future__ import annotations

import math

import numpy as np

from ipu_harness.packing import RowMajor
from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES

__all__ = ["gate_and_count", "padded_scores", "padding_value"]


def padding_value(plan: CasePlan) -> np.float32:
    """``tau - 800 / T`` in FP32, from the case's ``threshold`` and ``temperature``."""
    tau = np.float32(plan.params["threshold"])
    t = np.float32(plan.params["temperature"])
    return np.float32(tau - np.float32(800) / t)


class PaddedScores(RowMajor):
    """``[count]``, one row per 128, the lanes past ``count`` ``tau - 800 / T``."""

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        rows = super().to_rows(a, plan)
        rows.reshape(-1)[a.shape[-1] :] = padding_value(plan)
        return rows


class GateAndCount:
    """``[count + 1]``: ``ceil(count / 128)`` rows of selected scores, then the count row."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        return math.ceil((shape[0] - 1) / LANES) + 1

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        count = a.shape[0] - 1
        rows = np.zeros((self.rows(a.shape, plan), LANES), dtype=np.float64)
        rows[:-1].reshape(-1)[:count] = a[:count]
        rows[-1, 0] = a[count]
        return rows

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        count = shape[0] - 1
        return np.concatenate((r[:-1].reshape(-1)[:count], r[-1, :1]))


padded_scores = PaddedScores()
gate_and_count = GateAndCount()
