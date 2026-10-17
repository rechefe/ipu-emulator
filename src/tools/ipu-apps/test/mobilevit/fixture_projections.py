"""Temporary adapter: the MobileViT chain tests run the projections from
``//kernels/projections`` (E07-02), which replaced the ``ipu_apps`` harnesses.

``projection(name)`` gives a kernel's fixed shape, as the old ``app`` modules'
``K``/``N_OUT``/``N_TG``/``N_TOK`` constants did; ``run_projection`` runs it on
given data and weights and returns the old harness's output file as an array:
the raw XMEM rows ``(N_STREAM, N_TG, N_OUT, LANES)``, the first ``n_tok`` lanes
valid. E07-24 ports the chain tests to the harness's chaining API and removes
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES


@dataclass(frozen=True)
class Projection:
    """A projection kernel's fixed shape (its registry.toml entry)."""

    name: str
    k: int
    n_out: int
    n_tg: int
    n_tok: int
    n_stream: int
    activation: str

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        """The data ``b``: ``(N_STREAM, N_TG, K, N_TOK)``."""
        return (self.n_stream, self.n_tg, self.k, self.n_tok)


def projection(name: str) -> Projection:
    """The projection kernel ``name`` (``proj_qkv_192_p4``, ...)."""
    manifest = default_registry().kernel(name)
    p = {key: decl.default for key, decl in manifest.params.items()} | dict(manifest.fixed)
    return Projection(
        name, p["k"], p["n_out"], p["n_tg"], p["n_tok"], p["n_streams"], p["activation"]
    )


def run_projection(name: str, d: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, int]:
    """Run projection ``name`` on data ``d`` ``(N_STREAM, N_TG, K, N_TOK)`` and
    weights ``w`` ``(N_OUT, K)``, in FP32 on a fresh model.

    Returns:
        The output region's rows ``(N_STREAM, N_TG, N_OUT, LANES)`` FP32, and
        the cycles.
    """
    shape = projection(name)
    step = Chain(ElementFormat.FP32).run(
        name, {}, {"b": np.asarray(d, dtype=np.float32), "a": np.asarray(w, dtype=np.float32)}
    )
    result = step.result
    assert result.halted and result.fault is None, result.describe()
    rows = np.frombuffer(result.raw_outputs["y"], dtype="<f4")
    return rows.reshape(shape.n_stream, shape.n_tg, shape.n_out, LANES).copy(), result.cycles
