"""Seam: LayerNorm into the transformer matmul that follows it, at L3, L4 and
L5 (E07-24, from ``test_seam_layernorm_matmul_xmem_direct{,_l3,_l5}.py``).

``layernorm_*`` stores ``y`` as ``x`` is laid out: one row per (channel,
token group), the first ``n_tok`` lanes valid. ``matmul_*_x128`` reads its
data ``b`` (pack ``channel_major``) as one row per (channel, token group),
in the same order. Both programs read their input through CR0, so each input
is pinned at row 0 and LayerNorm's output cannot sit where the matmul reads:
the host copies the stored rows to row 0 verbatim (padding lanes included,
no repack), and the matmul reads them there. That copy is the seam under
test, as the old tests' byte copy was:

* through the real kernels the matmul gives the shared references chained in
  float64 (``ops.layernorm``, then ``ops.linear``), each step held to its own
  golden and the matmul's output poisoned first;
* one row of the copy set to 999 moves the result (the check has teeth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ipu_harness.chain import Chain
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference import ops

FP32 = ElementFormat.FP32
POISON = 1e3


@dataclass(frozen=True)
class Seam:
    """A LayerNorm entry feeding a matmul entry of the same layer.

    Attributes:
        layernorm: The producer.
        matmul: The consumer.
        channels: LayerNorm's channels, the matmul's ``k``.
        tokens: The tokens (two groups of 128 at L3).
        n_out: The matmul's output channels.
        seed: The old clean test's RandomState seed.
        mutation_seed: The old mutation test's seed.
        mutated_row: The stored row the old mutation test set to 999.
    """

    layernorm: str
    matmul: str
    channels: int
    tokens: int
    n_out: int
    seed: int
    mutation_seed: int
    mutated_row: int

    @property
    def groups(self) -> int:
        return -(-self.tokens // LANES)

    @property
    def group_tokens(self) -> int:
        return self.tokens // self.groups


SEAMS = {
    "L3": Seam(
        "layernorm_256x144", "matmul_144x144_x128", 144, 256, 144, 0x1EC4, 0x1EC3, 5 * 2 + 1
    ),
    "L4": Seam("layernorm_64x192", "matmul_192x192_x128", 192, 64, 192, 0x1EB0, 0x1EAF, 5),
    "L5": Seam("layernorm_16x240", "matmul_240x240_x128", 240, 16, 240, 0x5EA6, 0x5EA5, 7),
}


def _inputs(seam: Seam, seed: int) -> tuple[np.ndarray, ...]:
    """x ``[channels, tokens]``, gamma, beta and W, in the old draw order."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, size=(seam.channels, seam.tokens)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, size=seam.channels).astype(np.float32)
    beta = rng.uniform(-0.5, 0.5, size=seam.channels).astype(np.float32)
    w = rng.uniform(-1.0, 1.0, size=(seam.n_out, seam.channels)).astype(np.float32)
    return x, gamma, beta, w


def _want(seam: Seam, x, gamma, beta, w) -> np.ndarray:
    """``[groups, n_out, group_tokens]``: ``ops.layernorm`` then ``ops.linear``, in float64."""
    f64 = [torch.from_numpy(a.astype(np.float64)) for a in (x, gamma, beta, w)]
    normed = ops.layernorm(f64[0], f64[1], f64[2], dim=0)["y"]  # [channels, tokens]
    data = normed.reshape(seam.channels, seam.groups, seam.group_tokens).transpose(0, 1)
    return ops.linear(f64[3], data)["y"].numpy()


def _run(seam: Seam, seed: int, *, mutate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The chain: LayerNorm, the verbatim copy to row 0, the matmul. Returns
    the matmul's output ``[groups, n_out, group_tokens]`` and the reference."""
    x, gamma, beta, w = _inputs(seam, seed)
    chain = Chain(FP32)
    normed = chain.run(seam.layernorm, {}, {"x": x, "gamma": gamma, "beta": beta}, check=True)
    assert normed.result.passed, normed.result.describe()
    rows = chain.read(normed.resident["y"])  # (channel, group) rows, as stored
    if mutate:
        rows[seam.mutated_row] = 999.0
    data = chain.write(rows, at=0, pack="channel_major")
    out = chain.write(np.full((seam.groups * seam.n_out, LANES), POISON))
    # The matmul's golden would follow the mutated data; only the chained reference fails.
    step = chain.run(seam.matmul, {}, {"b": data, "a": w}, check=not mutate, at={"y": out})
    assert step.result.passed, step.result.describe()
    stored = chain.read(step.resident["y"])[:, : seam.group_tokens]
    assert not np.any(stored == POISON), "the matmul left output rows unwritten"
    got = stored.reshape(seam.groups, seam.n_out, seam.group_tokens)
    return got, _want(seam, x, gamma, beta, w)


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_layernorm_rows_feed_the_matmul_verbatim(seam: Seam):
    got, want = _run(seam, seam.seed)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_a_wrong_row_in_the_copy_moves_the_result(seam: Seam):
    got, want = _run(seam, seam.mutation_seed, mutate=True)
    assert float(np.max(np.abs(got - want))) > 1.0
