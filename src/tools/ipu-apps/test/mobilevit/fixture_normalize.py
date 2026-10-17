"""Temporary adapter: the MobileViT chain, seam and packed tests run the
LayerNorms from ``//kernels/normalize`` (E07-10), which replaced the
``ipu_apps`` harnesses.

``layernorm(name)`` gives an entry's fixed shape under the old ``app``
modules' constant names (``N_CH``, ``N_TG``, ``N_TPG``, ``N_TOK``,
``ROW_BYTES``); ``run_layernorm_rows`` writes ``x`` into XMEM row for row, as
the old harness staged its input file verbatim (padding lanes included), runs
the kernel on it with its output rows poisoned first, and returns its output
rows, cycles and instruction count. E07-24 ports the chain tests to the
harness's chaining API and removes this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipu_emu.reference import RunState
from ipu_harness.cases import expand
from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_harness.runner import SharedModel, stage
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES


@dataclass(frozen=True)
class LayerNorm:
    """A LayerNorm entry's fixed shape (its registry.toml entry)."""

    name: str
    N_CH: int
    tokens: int
    LANES: int = LANES
    ROW_BYTES: int = LANES * 4

    @property
    def N_TG(self) -> int:  # noqa: N802 - the old module constant's name
        return -(-self.tokens // LANES)

    @property
    def N_TPG(self) -> int:  # noqa: N802
        return self.tokens // self.N_TG

    @property
    def N_TOK(self) -> int:  # noqa: N802 - tokens per row, as the one-group modules had it
        return self.N_TPG

    @property
    def N_ROWS(self) -> int:  # noqa: N802
        return self.N_CH * self.N_TG


@dataclass(frozen=True)
class LayerNormRun:
    """One run: the output rows ``(N_ROWS, LANES)`` FP32, the cycles, and the
    instructions executed (every non-NOP op but the LR slot's, as the packed
    tests' ``count_instructions`` counts them)."""

    rows: np.ndarray
    cycles: int
    instructions: int


def layernorm(name: str) -> LayerNorm:
    """The LayerNorm entry ``name`` (``layernorm_64x192``, ...)."""
    fixed = default_registry().kernel(name).fixed
    return LayerNorm(name, fixed["channels"], fixed["tokens"])


def run_layernorm_rows(
    name: str,
    x_rows: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    poison: float = 1e3,
) -> LayerNormRun:
    """Run ``name`` in FP32 on a fresh model.

    Args:
        name: The entry.
        x_rows: ``(N_ROWS, LANES)`` FP32, row ``ch * N_TG + g``, written into
            XMEM at row 0 as it is.
        gamma, beta: ``[N_CH]``.
        poison: The output rows hold it before the run; a row still all
            ``poison`` after it fails the run.
    """
    shape = layernorm(name)
    manifest = default_registry().kernel(name)
    (case,) = (c for c in expand(manifest) if c.name == "default")
    chain = Chain(ElementFormat.FP32)
    chain.write(np.asarray(x_rows, dtype=np.float32), at=0)  # the program reads x through CR0
    inputs = {"gamma": np.asarray(gamma, np.float32), "beta": np.asarray(beta, np.float32)}
    shared = SharedModel(chain.model, inputs, {"x": 0})
    staged = stage(manifest, case, ElementFormat.FP32, shared=shared)
    model = staged.model
    start, size = staged.region_bytes("y")
    model.write_xmem(start, np.full(size // 4, poison, dtype="<f4").tobytes())
    instructions = cycles = 0
    while True:
        record = model.step()
        cycles += 1
        instructions += sum(op.slot != "lr" for op in record.slots)
        if model.status.state is not RunState.RUNNING:
            break
    assert model.status.fault is None, model.status
    rows = np.frombuffer(model.read_xmem(start, size), dtype="<f4").reshape(shape.N_ROWS, LANES)
    assert not np.all(rows == np.float32(poison), axis=1).any(), (
        f"{name} left output rows untouched"
    )
    return LayerNormRun(rows.copy(), cycles, instructions)
