"""Seam: the QKV projection (all 4 streams in one step) into the key-major
score kernel, at L4 and L5 (E07-24, from ``test_seam_proj_qkv_to_scores_km_repack.py``).

``proj_qkv_*_p4`` stores each output channel as a whole row with its first
``n_tok`` lanes valid, a stream's ``3 * n_head * d`` channels after the
previous stream's. ``attn_scores_km_*`` takes one head's Q channel-major
(``b``) and its K key-major (``a``). No base or stride makes the one the
other, so the host moves the data between the steps in two parts:

* ``repack`` takes each row's first ``n_tok`` lanes in the order the
  projection stored them (no reordering). On data whose every element
  encodes its own (stream, head, channel, token), it moves no element
  anywhere but where those indices say.
* ``key_major`` turns K's channel rows into key rows, the transpose the old
  harness did while staging K. The numeric tests cover it.

Through the real kernels the scores equal the shared references chained in
float64 (``ops.linear``, then ``ops.matmul``) of the projection's own weights
and data, with the score output poisoned first. The projection step is held
to its own golden; a score step to its datapath mirror, bit for bit
(``acc_fold``): the projection's outputs are larger than a case's
uniform(-1, 1) data, and their cancelling sums put a score past FP32_ORDER's
1e-6 floor, as on the attention family's own seams. And one wrong element in the repacked Q moves the scores
(the check has teeth). The score kernel's ``head`` parameter only names a
case's generated head: the host selects the head here.
"""

from __future__ import annotations

import numpy as np
import torch

from ipu_harness.chain import Chain
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference import ops
from ipu_kernels.reference.datapath import acc_fold

FP32 = ElementFormat.FP32
POISON = 1e3


def _fixed(name: str) -> dict:
    """A registry entry's parameter values: its fixed ones over the template's defaults."""
    manifest = default_registry().kernel(name)
    return {k: decl.default for k, decl in manifest.params.items()} | dict(manifest.fixed)


def repack(rows: np.ndarray, *, n_head: int, d: int, n_tok: int, first: int) -> np.ndarray:
    """``[stream, head, channel, token]`` from the projection's stored rows.

    ``rows`` is ``[stream, n_out, LANES]``: channel ``c`` of head ``h`` at row
    ``first + h * d + c``. Takes the ``n_head * d`` rows from ``first`` and
    their first ``n_tok`` lanes, in stored order: no reordering.
    """
    streams = rows.shape[0]
    return rows[:, first : first + n_head * d, :n_tok].reshape(streams, n_head, d, n_tok)


def key_major(k: np.ndarray) -> np.ndarray:
    """K ``[.., channel, key]`` as the score kernel takes it: ``[.., key, channel]``."""
    return np.ascontiguousarray(np.swapaxes(k, -1, -2))


def _qkv(chain: Chain, name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The projection step on uniform(-1, 1) data (the old seed and draw
    order: W, then each stream's data). Returns its stored rows ``[stream,
    n_out, LANES]``, W and the data ``[stream, k, n_tok]``."""
    p = _fixed(name)
    rng = np.random.RandomState(seed)
    w = rng.uniform(-1.0, 1.0, size=(p["n_out"], p["k"])).astype(np.float32)
    d = np.stack(
        [
            rng.uniform(-1.0, 1.0, size=(p["k"], p["n_tok"])).astype(np.float32)
            for _ in range(p["n_streams"])
        ]
    )
    step = chain.run(name, {}, {"b": d[:, None], "a": w}, check=True)  # n_tg = 1
    assert step.result.passed, step.result.describe()
    rows = chain.read(step.resident["y"]).reshape(p["n_streams"], p["n_out"], LANES)
    return rows, w, d


def _scores(chain: Chain, name: str, params: dict, q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """One score step on Q and K ``[.., d, n_tok]`` (channel-major), held to
    its datapath mirror, its output region poisoned first. Returns the stored rows'
    first ``n_tok`` lanes ``[.., n_tok (key), n_tok (query)]``."""
    n_tok = params["n_tok"]
    rows = int(np.prod(q.shape[:-2])) * n_tok
    # attn_scores_km_16x60 reads b through CR0, so b is pinned at row 0. The
    # guard is placed first fit, so it takes row 0 while it is free, and keeps
    # the poisoned output off b's rows.
    guard = chain.reserve(int(np.prod(q.shape[:-1])))
    out = chain.write(np.full((rows, LANES), POISON), pack="padded")
    chain.release(guard)
    a = key_major(k)
    step = chain.run(name, params, {"b": q, "a": a}, at={"y": out})
    assert step.result.passed, step.result.describe()  # halted, no fault
    stored = chain.read(step.resident["y"])
    assert not np.any(stored[:, :n_tok] == POISON), "the score kernel left rows unwritten"
    got = stored[:, :n_tok].reshape(*q.shape[:-2], n_tok, n_tok).astype(np.float32)
    a3, b3 = a.reshape(-1, *a.shape[-2:]), q.reshape(-1, *q.shape[-2:])
    mirror = acc_fold(np.swapaxes(b3, -1, -2), a3).reshape(got.shape)  # y = a @ b, ACC.ADD
    np.testing.assert_array_equal(got, mirror)
    return got


def _want(w: np.ndarray, d: np.ndarray, head: int, first_k: int, head_d: int) -> np.ndarray:
    """One head's key-major scores ``[stream, key, query]``: the shared
    references chained in float64, ``ops.linear`` (the projection) then
    ``ops.matmul`` (the scores, ``K^T Q``)."""
    c = ops.linear(torch.from_numpy(w.astype(np.float64)), torch.from_numpy(d.astype(np.float64)))
    qkv = c["y"]  # [stream, n_out, n_tok]
    q = qkv[:, head * head_d : (head + 1) * head_d]
    k = qkv[:, first_k + head * head_d : first_k + (head + 1) * head_d]
    return ops.matmul(k.transpose(-1, -2), q)["y"].numpy()


def test_repack_moves_each_element_only_where_its_indices_say():
    """Each element encodes its (stream, head, channel, token); padding lanes
    hold 0, which no encoded element is. An axis swap, a wrong stream order
    or a transposed token axis puts some element at the wrong index."""
    streams, n_head, d, n_tok, n_out = 4, 4, 48, 64, 576
    p, h, c, t = np.meshgrid(*map(np.arange, (streams, n_head, d, n_tok)), indexing="ij")
    encoded = (1 + p * 10_000 + h * 1_000 + c * 10 + t).astype(np.float32)
    rows = np.zeros((streams, n_out, LANES), np.float32)
    rows[:, : n_head * d, :n_tok] = encoded.reshape(streams, n_head * d, n_tok)
    got = repack(rows, n_head=n_head, d=d, n_tok=n_tok, first=0)
    np.testing.assert_array_equal(got, encoded)


def _l4(seed: int, head: int, *, corrupt: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """L4's chain on one head: the scores and their float64 reference;
    ``corrupt`` puts 999 in one channel of stream 1's repacked Q."""
    n_head, d, n_tok = 4, 48, 64
    chain = Chain(FP32)
    rows, w, data = _qkv(chain, "proj_qkv_192_p4", seed)
    q = repack(rows, n_head=n_head, d=d, n_tok=n_tok, first=0)[:, head]
    k = repack(rows, n_head=n_head, d=d, n_tok=n_tok, first=n_head * d)[:, head]
    if corrupt:
        q[1, 3, :] = 999.0
    # The score step's mirror follows the corrupted Q; only the chained reference fails.
    got = _scores(chain, "attn_scores_km_64x48", {"n_tok": n_tok, "d": d, "head": head}, q, k)
    return got, _want(w, data, head, n_head * d, d)


def test_proj_qkv_192_p4_feeds_attn_scores_km_64x48():
    got, want = _l4(0xF00D4, head=1)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2)


def test_a_wrong_repacked_element_moves_the_scores():
    got, want = _l4(0xBAD5EED, head=2, corrupt=True)
    assert float(np.max(np.abs(got - want))) > 1.0


def test_proj_qkv_240_p4_feeds_attn_scores_km_16x60():
    """L5's score kernel takes one stream per step: four steps, one per
    stream, each on that stream's repacked Q and K."""
    head, n_head, d, n_tok = 0, 4, 60, 16
    chain = Chain(FP32)
    rows, w, data = _qkv(chain, "proj_qkv_240_p4", 0xF00D5)
    q = repack(rows, n_head=n_head, d=d, n_tok=n_tok, first=0)[:, head]
    k = repack(rows, n_head=n_head, d=d, n_tok=n_tok, first=n_head * d)[:, head]
    want = _want(w, data, head, n_head * d, d)
    params = {"n_tok": n_tok, "d": d, "head": head}
    for stream in range(rows.shape[0]):
        got = _scores(chain, "attn_scores_km_16x60", params, q[stream], k[stream])
        np.testing.assert_allclose(got, want[stream], rtol=2e-3, atol=2e-2)
