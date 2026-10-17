"""The datapath mirrors (``ipu_kernels.reference.datapath``): each rounds where
its datapath rounds, and nowhere else."""

from __future__ import annotations

import numpy as np

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
    """The small products in a later chunk are added to a rounded partial."""
    P, V = _cancelling(steps=126)  # 128 products: one chunk
    two_chunks = np.concatenate([P, np.full((1, 1, 2), 2.0**-30, dtype=np.float32)], axis=-1)
    ones = np.ones((1, 1, two_chunks.shape[-1]), dtype=np.float32)
    first = agg_fold(P, V)[0, 0, 0]
    assert first == np.float32(126 * 2.0**-30)
    expected = np.float32(np.float64(first) + np.float64(np.float32(2 * 2.0**-30)))
    assert agg_fold(two_chunks, ones)[0, 0, 0] == expected
