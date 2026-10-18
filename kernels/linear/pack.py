"""The linear family's weight layout (E07-21).

``transposed`` lays the logical weights ``w``, ``(out_features, in_features)``
with ``in_features <= 128``, out as one XMEM row per input feature: row ``i``
holds column ``i`` of ``w`` in its first ``out_features`` lanes, zeros after.
``fully_connected`` loads row ``i`` into R_CYCLIC and multiplies it by input
element ``i``, so the accumulator collects ``y[s, :] = w @ x[s, :]``.
"""

import numpy as np

from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES


class Transposed:
    """``(out_features, in_features)`` as ``in_features`` rows of ``out_features`` lanes."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """One row per input feature.

        Raises:
            ValueError: ``shape`` is not 2-D, or the outputs do not fit one row.
        """
        if len(shape) != 2 or not 0 < shape[0] <= LANES:
            raise ValueError(f"transposed: {shape} must be (out_features <= {LANES}, in_features)")
        return shape[1]

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        """``a.T``, each row zero-padded to the lanes."""
        out = np.zeros((a.shape[1], LANES), dtype=np.float64)
        out[:, : a.shape[0]] = a.T
        return out

    def from_rows(self, r: np.ndarray, plan: CasePlan, shape: tuple[int, ...]) -> np.ndarray:
        """The ``(out_features, in_features)`` weights ``r`` holds (inverse of :meth:`to_rows`)."""
        return r.reshape(shape[1], LANES)[:, : shape[0]].T


transposed = Transposed()
