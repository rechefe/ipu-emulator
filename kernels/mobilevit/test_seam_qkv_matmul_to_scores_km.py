"""Seam: the single-stream QKV matmul into the key-major score kernel, at L4
(E07-24, from ``test_seam_qkv_matmul_scores_km_64x48.py``).

``matmul_576x192_x128`` runs once per stream and stores its 576 output
channels as one row each (Q's 4 heads x 48, then K's, then V's), the first
64 lanes valid. The old score harness read Q and K from tightly packed files
(64 elements per channel), so the old test found the two pitches disagreed
and a padding strip was unavoidable. ``attn_scores_km_64x48`` now takes Q
(``b``, ``[stream, d, n_tok]``) as one row per (stream, channel) too: the
pitches agree, and the host hands Q over by row selection alone, every lane
of each selected row as stored. K is still turned into key rows on the host
(the kernel takes it key-major, ``a``).

Through the real kernels the scores equal the shared references chained in
float64 (``ops.linear`` per stream, then ``ops.matmul``), with the score
output poisoned first. Each matmul step is held to its own golden; the score
step to ``acc_fold`` bit for bit (chained, cancelling sums exceed
FP32_ORDER's 1e-6 floor, as on the other score seams).
"""

from __future__ import annotations

import numpy as np
import torch

from ipu_harness.chain import Chain
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference import ops
from ipu_kernels.reference.datapath import acc_fold

FP32 = ElementFormat.FP32
POISON = 1e3
STREAMS, HEADS, D, N_TOK, K, N_OUT = 4, 4, 48, 64, 192, 576
Q_ROW0, K_ROW0 = 0, HEADS * D  # Q's channels, then K's, in each stream's output


def _streams(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """W ``[576, 192]`` and each stream's data ``[stream, 192, 64]``, in the old draw order."""
    rng = np.random.RandomState(seed)
    w = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)
    d = np.stack(
        [rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32) for _ in range(STREAMS)]
    )
    return w, d


def _want(w: np.ndarray, d: np.ndarray, head: int) -> np.ndarray:
    """``[stream, key, query]``: ``ops.linear`` per stream, then ``ops.matmul`` (``K^T Q``)."""
    qkv = ops.linear(torch.from_numpy(w.astype(np.float64)), torch.from_numpy(d.astype(np.float64)))
    c = qkv["y"]  # [stream, 576, 64]
    q = c[:, Q_ROW0 + head * D : Q_ROW0 + (head + 1) * D]
    k = c[:, K_ROW0 + head * D : K_ROW0 + (head + 1) * D]
    return ops.matmul(k.transpose(-1, -2), q)["y"].numpy()


def test_qkv_matmul_rows_feed_attn_scores_km_64x48_by_row_selection():
    w, d = _streams(0x9EC1)
    head = 1
    chain = Chain(FP32)
    stored = []
    for stream in range(STREAMS):
        step = chain.run("matmul_576x192_x128", {}, {"b": d[stream][None], "a": w}, check=True)
        assert step.result.passed, step.result.describe()
        stored.append(chain.read(step.resident["y"]))  # [576, LANES]: every lane as stored
    rows = np.stack(stored)

    # Q: the head's rows of each stream, every lane as stored, as one
    # (stream, channel) row each: the layout b takes, so it goes in as rows.
    q_rows = rows[:, Q_ROW0 + head * D : Q_ROW0 + (head + 1) * D].reshape(STREAMS * D, LANES)
    q = chain.write(q_rows)
    k = rows[:, K_ROW0 + head * D : K_ROW0 + (head + 1) * D, :N_TOK]  # [stream, d, key]
    a = np.ascontiguousarray(np.swapaxes(k, -1, -2))  # key-major, as a is declared
    out = chain.write(np.full((STREAMS * N_TOK, LANES), POISON), pack="padded")
    params = {"n_tok": N_TOK, "d": D, "head": head}
    step = chain.run("attn_scores_km_64x48", params, {"b": q, "a": a}, at={"y": out})
    assert step.result.passed, step.result.describe()  # halted, no fault

    scores = chain.read(step.resident["y"])[:, :N_TOK]
    assert not np.any(scores == POISON), "the score kernel left rows unwritten"
    got = scores.reshape(STREAMS, N_TOK, N_TOK).astype(np.float32)
    q_values = q_rows[:, :N_TOK].reshape(STREAMS, D, N_TOK).astype(np.float32)
    np.testing.assert_array_equal(got, acc_fold(np.swapaxes(q_values, -1, -2), a))
    np.testing.assert_allclose(got, _want(w, d, head), rtol=2e-3, atol=2e-2)
