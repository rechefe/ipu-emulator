"""The datapath mirrors (``ipu_kernels.reference.datapath``): each rounds where
its datapath rounds, and nowhere else."""

from __future__ import annotations

import numpy as np
import pytest

from ipu_isa.geometry import LANES
from ipu_kernels.reference.datapath import acc_fold, agg_fold


def _cancelling(steps: int) -> tuple[np.ndarray, np.ndarray]:
    """One lane, one output: products 1, 2**-30 (repeated), -1, in that order."""
    products = np.array([1.0, *([2.0**-30] * steps), -1.0], dtype=np.float32)
    return products[None, None, :], np.ones((1, 1, products.size), dtype=np.float32)


def test_both_folds_are_the_matmul_when_nothing_rounds():
    rng = np.random.default_rng(0)
    P = rng.integers(-8, 8, size=(2, 5, 300)).astype(np.float32)
    V = rng.integers(-8, 8, size=(2, 3, 300)).astype(np.float32)
    exact = np.einsum("bis,bts->bti", P, V)
    np.testing.assert_array_equal(acc_fold(P, V), exact)
    np.testing.assert_array_equal(agg_fold(P, V), exact)


def test_acc_rounds_every_step_and_agg_once_per_chunk():
    """1 + 2**-30 is 1 in float32, so a float32 running sum loses each small
    product; AGG's float64 fold keeps them until its one rounding."""
    P, V = _cancelling(steps=4)
    assert acc_fold(P, V)[0, 0, 0] == 0.0
    assert agg_fold(P, V)[0, 0, 0] == np.float32(4 * 2.0**-30)


def test_agg_rounds_each_128_lane_chunk_before_adding_the_next():
    """Chunk 0 is 1 + 127 * 2**-40, which rounds to 1 in float32; chunk 1
    takes it back off. One float64 fold over both chunks would keep the
    small products; AGG's per-chunk rounding loses them."""
    products = np.array([1.0, *([2.0**-40] * 127), -1.0], dtype=np.float32)
    P = products[None, None, :]
    V = np.ones_like(P)
    assert np.float32(1.0 + 127 * 2.0**-40) == 1.0  # the premise
    assert agg_fold(P, V)[0, 0, 0] == 0.0
    assert np.float32(products.astype(np.float64).sum()) != 0.0  # a single fold differs


def test_agg_adds_a_later_chunk_unrounded_to_the_partial():
    """Chunk 0 stores 1. Chunk 1's fold is 2**-24 + 2**-50, which rounds to
    2**-24 in float32, and 1 + 2**-24 is a tie that rounds to 1. AGG.SUM adds
    the unrounded fold, which is past the tie: 1 + 2**-23."""
    products = np.zeros(LANES + 2, dtype=np.float32)
    products[0], products[LANES], products[LANES + 1] = 1.0, 2.0**-24, 2.0**-50
    P = products[None, None, :]
    V = np.ones_like(P)
    assert np.float32(1.0) + np.float32(2.0**-24 + 2.0**-50) == 1.0  # rounded first: lost
    assert agg_fold(P, V)[0, 0, 0] == np.float32(1.0 + 2.0**-23)


def test_an_empty_contraction_is_refused():
    empty = np.zeros((1, 1, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="empty contraction"):
        acc_fold(empty, empty)
    with pytest.raises(ValueError, match="empty contraction"):
        agg_fold(empty, empty)
