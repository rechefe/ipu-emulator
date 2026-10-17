"""Temporary adapter: the MobileViT seam tests run unfold from
``//kernels/reshape`` (E07-05), which replaced the ``ipu_apps`` harnesses.

``unfold(name)`` gives a kernel's fixed shape under the old ``app`` modules'
constant names (``C``, ``H``, ``W``, ``N_TOK``, ...); ``run_unfold`` runs it on
a ``[C, H, W]`` tensor, its output rows poisoned first, and returns the raw
rows the old harness left in XMEM: the four streams one after another, the
first ``N_TOK`` lanes of each row valid. E07-24 ports the chain tests to the
harness's chaining API and removes this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES

#: Where the output rows go: well clear of x and its ones row.
_OUTPUT_ROW = 4096


@dataclass(frozen=True)
class Unfold:
    """An unfold kernel's fixed shape (its kernel.toml), named as the old module's constants."""

    name: str
    H: int
    W: int
    C: int
    N_STREAMS: int = 4
    LANES: int = LANES

    @property
    def N_OUT(self) -> int:  # noqa: N802 - the old module constant's name
        return self.C

    @property
    def N_TOK(self) -> int:  # noqa: N802
        return (self.H // 2) * (self.W // 2)

    @property
    def N_STRIPES(self) -> int:  # noqa: N802
        return -(-self.H * self.W // LANES)


def unfold(name: str) -> Unfold:
    """The unfold kernel ``name`` (``unfold_16x16x192``, ...)."""
    params = default_registry().kernel(name).params
    return Unfold(name, params["h"].default, params["w"].default, params["c"].default)


def run_unfold(name: str, x: np.ndarray, poison: float) -> tuple[bytes, int]:
    """Run unfold ``name`` on ``x`` ``[C, H, W]`` in FP32, its output rows
    holding ``poison`` first.

    Returns:
        The output rows' bytes (stream after stream, ``C`` rows each for a
        tensor whose phases fit one row), and the cycles.
    """
    shape = unfold(name)
    rows = shape.N_STREAMS * shape.C * -(-shape.N_TOK // LANES)
    chain = Chain(ElementFormat.FP32)
    chain.write(np.full((rows, LANES), poison), at=_OUTPUT_ROW)
    step = chain.run(name, {}, {"x": np.asarray(x, dtype=np.float32)}, at={"y": _OUTPUT_ROW})
    result = step.result
    assert result.halted and result.fault is None, result.describe()
    return result.raw_outputs["y"], result.cycles
