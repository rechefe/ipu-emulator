"""Unit tests for the host-side softmax/fold TEST FIXTURES (not kernels).

The property that matters most for these stubs: they must PRESERVE the
layout they are handed. A softmax that silently transposed its input would
still produce plausible-looking numbers and could hide a real incompatibility
between a score kernel and attn@V -- these tests pin that down directly,
independent of the full-layer chain tests that use the stub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Makes this file's own directory importable whether test/ is a package
# (bazel auto-generates test/__init__.py) or a flat rootdir (plain
# pytest/uv run pytest, no __init__.py) -- see the matching comment in
# test_full_layer_l4.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixture_host_softmax_fold_stubs import softmax_query_major, softmax_key_major, fold_stub


def test_softmax_query_major_rows_sum_to_one() -> None:
    rng = np.random.RandomState(0)
    scores = rng.uniform(-5, 5, size=(4, 16, 16))  # [head, query, key]
    probs = softmax_query_major(scores)
    assert probs.shape == scores.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-10)
    assert np.all(probs >= 0)


def test_softmax_key_major_columns_sum_to_one() -> None:
    """Key-major storage: row s = key s, column q = query q. A correct
    softmax still normalizes over KEYS, which here is the ROW axis -- so each
    COLUMN (fixed query, varying key/row) must sum to 1, not each row."""
    rng = np.random.RandomState(1)
    scores_km = rng.uniform(-5, 5, size=(4, 16, 16))  # [head, key, query]
    probs = softmax_key_major(scores_km)
    assert probs.shape == scores_km.shape
    np.testing.assert_allclose(probs.sum(axis=-2), 1.0, rtol=1e-10)
    assert np.all(probs >= 0)


def test_softmax_query_major_and_key_major_agree_up_to_transpose() -> None:
    """The two stubs must compute the SAME softmax, differing only in which
    axis is "rows" -- softmax_key_major(S.T) == softmax_query_major(S).T.
    If they disagreed here, one of the two chains' softmax convention would
    be wrong relative to the other, exactly the class of bug a transposing
    stub would hide.
    """
    rng = np.random.RandomState(2)
    scores_qm = rng.uniform(-3, 3, size=(16, 16))  # [query, key]
    probs_qm = softmax_query_major(scores_qm)

    scores_km = scores_qm.T  # [key, query]
    probs_km = softmax_key_major(scores_km)

    np.testing.assert_allclose(probs_km, probs_qm.T, rtol=1e-12)


def test_softmax_query_major_does_not_transpose() -> None:
    """Construct an asymmetric input where transposing would change values
    at almost every index, then assert the stub's output shape/axis-order
    still matches the INPUT's row axis (query), not the key axis."""
    scores = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 10.0, -10.0],
    ])
    probs = softmax_query_major(scores)
    # Row 0 should be dominated by column 0 (query 0 attends mostly to key 0).
    assert probs[0, 0] > 0.99
    # Row 2 should be dominated by column 1, NOT column 0 or 2.
    assert probs[2, 1] > 0.99
    # If the stub had transposed, row/column roles would swap and this would fail:
    # e.g. probs[0, 2] (would-be transposed hot spot) must stay near zero.
    assert probs[0, 2] < 0.01


def test_fold_stub_reduces_last_axis_and_preserves_leading_shape() -> None:
    x = np.arange(2 * 3 * 8).reshape(2, 3, 8).astype(np.float64)
    out = fold_stub(x, out_channels=4)
    assert out.shape == (2, 3, 4)
    # Group of 2 elements averaged: channel c = mean(x[..., 2c:2c+2]).
    expected = x.reshape(2, 3, 4, 2).mean(axis=-1)
    np.testing.assert_allclose(out, expected)
