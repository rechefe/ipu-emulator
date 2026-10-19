"""Seam: unfold into the QKV matmul, at L4 and L5 (E07-24, from
``test_seam_unfold_matmul_xmem_direct_l{4,5}.py``).

``unfold_*`` stores each of its four phases (streams) as one row per channel,
the first ``n_tok`` lanes valid; what the other lanes hold is the kernel's
business (stale R_ACC at L4, zero at L5). ``matmul_*_x128`` reads its data
``b`` as one row per channel, in the same order, and contracts each lane on
its own: lanes past ``n_tok`` reach only output lanes past ``n_tok``. Both
programs read their input through CR0, so the host copies one stream's
stored rows to row 0 verbatim (every lane, no repack), and the matmul reads
them there. That copy is the seam under test:

* through the real kernels the matmul gives the shared references chained in
  float64 (the stride-2 decimation, then ``ops.linear``), each step held to
  its own golden and the matmul's output poisoned first;
* 999 in every row's lanes past ``n_tok`` leaves the valid output as it is
  (the lanes are inert, whatever unfold leaves there);
* 999 in one channel's valid lanes moves the result (the check has teeth).
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
STREAM = 0  # phase (0, 0): the one stream the old tests fed through


@dataclass(frozen=True)
class Seam:
    """An unfold entry feeding a matmul entry of the same layer.

    Attributes:
        unfold: The producer.
        matmul: The consumer.
        c: Channels, the matmul's ``k``.
        hw: The spatial height and width.
        n_out: The matmul's output channels.
        seed: The old clean test's RandomState seed.
        mutation_seed: The old mutation test's seed.
        mutated_channel: The channel whose valid lanes the old mutation test set to 999.
        tail_seed: The old tail-poison test's seed (L4 only had one; L5 uses it too).
    """

    unfold: str
    matmul: str
    c: int
    hw: int
    n_out: int
    seed: int
    mutation_seed: int
    mutated_channel: int
    tail_seed: int

    @property
    def n_tok(self) -> int:
        return (self.hw // 2) ** 2


SEAMS = {
    "L4": Seam("unfold_16x16x192", "matmul_576x192_x128", 192, 16, 576, 0x4A12, 0x4A11, 9, 0x4A13),
    "L5": Seam("unfold_8x8x240", "matmul_720x240_x128", 240, 8, 720, 0x5E16, 0x5E15, 13, 0x4A13),
}


def _inputs(seam: Seam, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """x ``[c, h, w]`` and W, in the old draw order."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(seam.c, seam.hw, seam.hw)).astype(np.float32)
    w = rng.uniform(-1.0, 1.0, size=(seam.n_out, seam.c)).astype(np.float32)
    return x, w


def _want(seam: Seam, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``[n_out, n_tok]``: stream ``STREAM``'s stride-2 decimation, then ``ops.linear``, in float64."""
    r, c = STREAM // 2, STREAM % 2
    data = x[:, r::2, c::2].reshape(1, seam.c, seam.n_tok).astype(np.float64)  # one token group
    return ops.linear(torch.from_numpy(w.astype(np.float64)), torch.from_numpy(data))["y"][
        0
    ].numpy()


def _run(seam: Seam, seed: int, *, valid=None, tail=None) -> tuple[np.ndarray, np.ndarray]:
    """The chain: unfold, stream ``STREAM``'s rows copied to row 0, the matmul.
    ``valid`` puts that value in one channel's valid lanes of the copy, ``tail``
    in every row's lanes past ``n_tok``. Returns the matmul's output
    ``[n_out, n_tok]`` and the reference."""
    x, w = _inputs(seam, seed)
    chain = Chain(FP32)
    unfolded = chain.run(seam.unfold, {}, {"x": x}, check=True)
    assert unfolded.result.passed, unfolded.result.describe()
    rows = chain.read(unfolded.resident["y"].slice(STREAM * seam.c, seam.c))
    if valid is not None:
        rows[seam.mutated_channel, : seam.n_tok] = valid
    if tail is not None:
        rows[:, seam.n_tok :] = tail
    data = chain.write(rows, at=0, pack="channel_major")
    out = chain.write(np.full((seam.n_out, LANES), POISON))
    # With a wrong valid lane the matmul's golden would follow the copy; only the chained reference fails.
    step = chain.run(seam.matmul, {}, {"b": data, "a": w}, check=valid is None, at={"y": out})
    assert step.result.passed, step.result.describe()
    stored = chain.read(step.resident["y"])[:, : seam.n_tok]
    assert not np.any(stored == POISON), "the matmul left output rows unwritten"
    return stored, _want(seam, x, w)


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_unfold_rows_feed_the_matmul_verbatim(seam: Seam):
    got, want = _run(seam, seam.seed)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_the_lanes_past_n_tok_are_inert(seam: Seam):
    got, want = _run(seam, seam.tail_seed, tail=999.0)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_a_wrong_valid_lane_in_the_copy_moves_the_result(seam: Seam):
    got, want = _run(seam, seam.mutation_seed, valid=999.0)
    assert float(np.max(np.abs(got - want))) > 1.0
