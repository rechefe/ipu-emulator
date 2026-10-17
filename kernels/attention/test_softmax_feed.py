"""The L5 score kernels feeding the softmax kernels (E07-12).

Can a score kernel's raw output (whole rows, n_tok live lanes) go straight
into softmax? The softmax kernels take exactly a ``(rows, n)`` (or
``(rows, width)``) input, so the raw rows are refused rather than read as
garbage; the scores (the live lanes, as ``y`` unpacks them) give the right
softmax, along each row of the query-major scores and down each column of the
key-major ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness.cases import at_values, expand
from ipu_harness.registry import default_registry
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES

FP32 = ElementFormat.FP32


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


def _run(name: str, values: dict, inputs: dict[str, np.ndarray], tmp: Path):
    """``name``'s default case at ``values``, on ``inputs`` (written as .npy)."""
    kernel = default_registry().kernel(name)
    default = next(c for c in expand(kernel) if c.name == "default")
    files = {}
    for tensor, array in inputs.items():
        files[tensor] = tmp / f"{name}_{tensor}.npy"
        np.save(files[tensor], array)
    case = at_values(default, f"feed[{name}]", values)
    return run_kernel(kernel, case, FP32, RunOptions(inputs=files))


def test_query_major_scores_feed_softmax_rows(tmp_path):
    n, d = 16, 60
    rng = np.random.RandomState(0)
    Q = (rng.randn(d, n) * 2).astype(np.float32)
    K = (rng.randn(d, n) * 2).astype(np.float32)
    scores = _run("qk_scores_16x60", {}, {"a": np.ascontiguousarray(Q.T), "b": K}, tmp_path)
    assert scores.passed, scores.describe()
    raw = np.frombuffer(scores.raw_outputs["y"], dtype="<f4").reshape(n, LANES)

    with pytest.raises(ConfigurationError, match="shape"):
        _run("softmax_rows_partial", {"rows": n, "n": n}, {"x": raw}, tmp_path)

    y = np.asarray(scores.outputs["y"], dtype=np.float32)
    got = _run("softmax_rows_partial", {"rows": n, "n": n}, {"x": y}, tmp_path)
    assert got.passed, got.describe()
    err = float(np.abs(np.asarray(got.outputs["y"]) - _softmax(Q.T @ K, axis=1)).max())
    assert err < 1e-4, f"softmax of the qk_scores_16x60 scores: max abs error {err:.3e}"


def test_key_major_scores_feed_softmax_columns(tmp_path):
    n, d, head, heads = 16, 60, 1, 4
    rng = np.random.RandomState(1)
    Q = rng.uniform(-1.0, 1.0, size=(heads * d, n)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(heads * d, n)).astype(np.float32)
    q, k = Q[head * d : (head + 1) * d], K[head * d : (head + 1) * d]  # [d, n]
    scores = _run(
        "attn_scores_km_16x60", {"head": head}, {"a": np.ascontiguousarray(k.T), "b": q}, tmp_path
    )
    assert scores.passed, scores.describe()
    raw = np.frombuffer(scores.raw_outputs["y"], dtype="<f4").reshape(n, LANES)

    with pytest.raises(ConfigurationError, match="shape"):
        _run("softmax_columns_packed", {"rows": n, "width": n}, {"x": raw}, tmp_path)

    # Key-major: row s is key s. Softmax reduces over the keys, down each
    # column of the [key, query] matrix.
    y = np.asarray(scores.outputs["y"], dtype=np.float32)
    got = _run("softmax_columns_packed", {"rows": n, "width": n}, {"x": y}, tmp_path)
    assert got.passed, got.describe()
    expected = _softmax((q.T @ k).T, axis=0)
    err = float(np.abs(np.asarray(got.outputs["y"]) - expected).max())
    assert err < 1e-4, f"softmax of the attn_scores_km_16x60 scores: max abs error {err:.3e}"
