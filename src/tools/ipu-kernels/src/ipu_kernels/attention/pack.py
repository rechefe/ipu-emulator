"""The attention family's pack transform (E07-12), found by name as
``ipu_kernels.attention.pack``.

``padded`` is the row-major default with the padding lanes set to the case's
``pad`` parameter instead of 0. A chained producer leaves another stream's
data in those lanes, not zeros; a case with ``pad`` != 0 shows the kernel
neither reads them into a live lane nor stores them (the old ``run_poked``
probes). At ``pad = 0`` it is the default transform, byte for byte.

The score kernels' outputs take it too (E07-13): unpacking is the row-major
default's, and a ``Chain`` hands a resident on only when its pack is the one
the consumer declares, so the scores feed ``attn_v``'s ``P`` in place.
"""

from __future__ import annotations

import numpy as np

from ipu_harness.packing import RowMajor
from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES


class Padded(RowMajor):
    """Row-major, the padding lanes of each row set to ``plan.params["pad"]``."""

    def to_rows(self, a: np.ndarray, plan: CasePlan) -> np.ndarray:
        """``a`` as ``(rows, 128)``, each last-dimension run padded with ``pad``."""
        width = -(-a.shape[-1] // LANES) * LANES
        fill = float(plan.params.get("pad", 0.0))
        padded = np.full((*a.shape[:-1], width), fill, dtype=np.float64)
        padded[..., : a.shape[-1]] = a
        return padded.reshape(-1, LANES)


padded = Padded()
