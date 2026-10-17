"""The projections' output transform (E07-02), shared by the family's templates,
and the packed linear's weight transform (E07-23).

``projection_output`` lays ``y``, ``(streams, n_tg, n_out, n_tok)``, out as
the kernels' ``ST.AAQ`` rows: grouped channel-major, row ``(p, tg, j)`` at
``p * n_tg * n_out + tg * n_out + j``, the first ``n_tok`` lanes valid. On
unpack the lanes past ``n_tok`` (what the store leaves there) are dropped.
That is the row-major default over the logical shape; it is named here so
``kernel.toml`` says what the store writes, as ``ipu_apps``' ``teardown`` and
``read_output`` did.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_harness.packing import RowMajor
from ipu_harness.params import CasePlan
from ipu_isa.geometry import LANES


class ProjectionOutput(RowMajor):
    """``(..., n_out, n_tok)`` with ``n_tok <= LANES``: one row per output
    channel of each (stream, token group), the first ``n_tok`` lanes valid."""

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        """One row per index of every dimension but the tokens.

        Raises:
            ValueError: The tokens do not fit one row.
        """
        if not 0 < shape[-1] <= LANES:
            raise ValueError(f"projection_output: n_tok ({shape[-1]}) must fit one row, 1..{LANES}")
        return super().rows(shape, plan)


projection_output = ProjectionOutput()


Array = npt.NDArray[Any]


class WeightChunks:
    """Output-major weights ``(n_out, k)`` as ``ceil(k / 128)`` chunks of
    ``n_out`` rows: row ``c * n_out + o`` holds ``a[o, 128 c : 128 c + 128]``,
    the last chunk zero-padded (packed_output_linear's weight rows)."""

    @staticmethod
    def _check(shape: tuple[int, ...]) -> None:
        if len(shape) != 2:
            raise ValueError(f"weight_chunks lays out (n_out, k), not {shape}")

    def rows(self, shape: tuple[int, ...], plan: CasePlan) -> int:
        self._check(shape)
        return shape[0] * math.ceil(shape[1] / LANES)

    def to_rows(self, a: Array, plan: CasePlan) -> Array:
        self._check(a.shape)
        n_out, k = a.shape
        chunks = math.ceil(k / LANES)
        padded = np.zeros((n_out, chunks * LANES), dtype=np.float64)
        padded[:, :k] = a
        return padded.reshape(n_out, chunks, LANES).swapaxes(0, 1).reshape(-1, LANES)

    def from_rows(self, r: Array, plan: CasePlan, shape: tuple[int, ...]) -> Array:
        self._check(shape)
        n_out, k = shape
        chunks = math.ceil(k / LANES)
        return r.reshape(chunks, n_out, LANES).swapaxes(0, 1).reshape(n_out, -1)[:, :k]


weight_chunks = WeightChunks()
