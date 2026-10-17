"""Temporary adapter: the MobileViT chain and seam tests run the residual adds
from ``//kernels/elementwise`` (E07-06), which replaced the ``ipu_apps``
harnesses.

``residual_add(name)`` gives an entry's fixed shape under the old ``app``
modules' constant names (``N_CH``, ``N_TOK``, ``N_ROWS``, ``ROW_BYTES``);
``run_residual_add_rows`` writes ``a`` and ``b`` into XMEM row for row, as the
old harness staged its input files verbatim (a producer's whole rows, stale
lanes included), runs the kernel on them and returns its output rows. E07-24
ports the chain tests to the harness's chaining API and removes this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES


@dataclass(frozen=True)
class ResidualAdd:
    """A residual-add entry's fixed shape (its registry.toml entry)."""

    name: str
    N_TOK: int
    N_CH: int
    LANES: int = LANES
    ROW_BYTES: int = LANES * 4

    @property
    def N_ROWS(self) -> int:  # noqa: N802 - the old module constant's name
        return self.N_CH * -(-self.N_TOK // LANES)


def residual_add(name: str) -> ResidualAdd:
    """The residual-add entry ``name`` (``residual_add_64x192``, ...)."""
    fixed = default_registry().kernel(name).fixed
    return ResidualAdd(name, fixed["tokens"], fixed["channels"])


def run_residual_add_rows(
    name: str, a_rows: np.ndarray, b_rows: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run ``name`` on ``a_rows`` and ``b_rows``, each ``(N_ROWS, LANES)`` FP32,
    written into XMEM as they are, in FP32 on a fresh model.

    Returns:
        The output region's rows ``(N_ROWS, LANES)`` FP32, and the cycles.
    """
    chain = Chain(ElementFormat.FP32)
    a = chain.write(np.asarray(a_rows, dtype=np.float32), at=0)  # the program reads a through CR0
    b = chain.write(np.asarray(b_rows, dtype=np.float32))
    result = chain.run(name, {}, {"a": a, "b": b}).result
    assert result.halted and result.fault is None, result.describe()
    rows = np.frombuffer(result.raw_outputs["y"], dtype="<f4")
    return rows.reshape(len(a_rows), LANES).copy(), result.cycles
