"""Seams outside the attention chains: OutProj into the residual add (L4,
L5), and LayerNorm into the QKV matmul with a chained producer's data in its
padding lanes (L4) (E07-24, from ``test_seam_pipeline_boundaries.py``).

Every program here reads its first input through CR0, so each input is
pinned at row 0 and a producer's output cannot sit where the consumer reads:
the host copies the stored rows to row 0 verbatim (every lane, no repack),
and the consumer reads them there. The old tests fed one harness's output
file to the next harness, and found two defects at that boundary: an L5
OutProj that cropped its rows (fixed since), and a LayerNorm whose full rows
the QKV matmul's file loader misread as tightly packed. Neither has a file
boundary left: rows go from one kernel's region to the next as stored.

Each consumer is held to the shared references chained in float64, each
step to its own golden, and each consumer's output is poisoned first.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ipu_harness.chain import Chain
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference import ops

FP32 = ElementFormat.FP32
POISON = 1e3


def _f64(*arrays: np.ndarray) -> list[torch.Tensor]:
    return [torch.from_numpy(a.astype(np.float64)) for a in arrays]


def _poisoned(chain: Chain, rows: int, pack: str | None = None):
    """An output region filled with poison, for a step's ``at``."""
    return chain.write(np.full((rows, LANES), POISON), pack=pack)


def _unwritten(stored: np.ndarray) -> bool:
    return bool(np.any(stored == POISON))


#: OutProj -> residual add: (matmul, residual add, channels, tokens, the old seed).
OUTPROJ = {
    "L5": ("matmul_240x240_x128", "residual_add_16x240", 240, 16, 0x5EAF1),
    "L4": ("matmul_192x192_x128", "residual_add_64x192", 192, 64, 0xC0FFEE),
}


@pytest.mark.parametrize("seam", OUTPROJ.values(), ids=OUTPROJ.keys())
def test_outproj_rows_feed_the_residual_add_verbatim(seam):
    """OutProj's stored rows, every lane, as the residual add's ``a``; ``b``
    is zero, so the sum is OutProj's own result."""
    matmul, residual_add, channels, tokens, seed = seam
    rng = np.random.RandomState(seed)
    d = rng.uniform(-1.0, 1.0, size=(channels, tokens)).astype(np.float32)
    w = rng.uniform(-1.0, 1.0, size=(channels, channels)).astype(np.float32)
    chain = Chain(FP32)
    projected = chain.run(matmul, {}, {"b": d[None], "a": w}, check=True)
    assert projected.result.passed, projected.result.describe()
    a = chain.write(chain.read(projected.resident["y"]), at=0)
    b = np.zeros((channels, tokens), np.float32)
    out = _poisoned(chain, channels)
    step = chain.run(residual_add, {}, {"a": a, "b": b}, check=True, at={"y": out})
    assert step.result.passed, step.result.describe()
    stored = chain.read(step.resident["y"])[:, :tokens]
    assert not _unwritten(stored), "the residual add left output rows unwritten"
    w64, d64, b64 = _f64(w, d[None], b)
    want = ops.residual_add(ops.linear(w64, d64)["y"][0], b64)["y"].numpy()
    np.testing.assert_allclose(stored, want, rtol=1e-4, atol=1e-3)


def test_layernorm_rows_feed_the_qkv_matmul_with_data_in_their_padding_lanes():
    """LayerNorm's input rows hold a chained producer's data (poison) past
    the 64 tokens, as its output rows then do; the QKV matmul reads the
    copy's first 64 lanes only, and gives the chained reference. (The old
    file loader read these full rows as tightly packed: the defect the old
    test recorded.)"""
    channels, tokens, n_out = 192, 64, 576
    rng = np.random.RandomState(0x14192)
    x = rng.uniform(-2.0, 2.0, size=(channels, tokens)).astype(np.float32)
    gamma = np.ones(channels, np.float32)
    beta = np.zeros(channels, np.float32)
    w = rng.uniform(-1.0, 1.0, size=(n_out, channels)).astype(np.float32)
    chain = Chain(FP32)
    x_rows = np.full((channels, LANES), POISON)
    x_rows[:, :tokens] = x
    staged = chain.write(x_rows, at=0)  # x is read through CR0: row 0, as declared
    normed = chain.run(
        "layernorm_64x192", {}, {"x": staged, "gamma": gamma, "beta": beta}, check=True
    )
    assert normed.result.passed, normed.result.describe()
    data = chain.write(chain.read(normed.resident["y"]), at=0, pack="channel_major")
    out = _poisoned(chain, n_out)
    step = chain.run("matmul_576x192_x128", {}, {"b": data, "a": w}, check=True, at={"y": out})
    assert step.result.passed, step.result.describe()
    stored = chain.read(step.resident["y"])[:, :tokens]
    assert not _unwritten(stored), "the matmul left output rows unwritten"
    x64, g64, b64, w64 = _f64(x, gamma, beta, w)
    normed64 = ops.layernorm(x64, g64, b64, dim=0)["y"]
    want = ops.linear(w64, normed64[None])["y"][0].numpy()
    np.testing.assert_allclose(stored, want, rtol=2e-3, atol=2e-2)
