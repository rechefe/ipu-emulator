"""The seams of the two attention chains (E07-13): score kernels into attn@V.

A kernel's own cases stage its inputs from numpy, so they cannot see a pitch,
base, extent or ordering mismatch between a producer and its consumer. Here
the scores never leave XMEM: on one ``Chain``, each head's score step writes
its rows straight into that head's block of the consumer's P (``at``, a slice
of a reserved tensor), and the attn@V step reads P where it lies.

Each seam is held to:

* each score step's own golden (``check=True``);
* a float64 reference of the whole two-stage computation, at the old seam
  tests' tolerances;
* its datapath mirror, bit for bit, on the P the consumer read (``agg_fold``
  for the query-major chain, ``acc_fold`` for the key-major one). The attn@V
  step is not held to its own tier: chained scores are larger than a case's
  uniform(-1, 1) P, and their cancelling sums put a K = 64 output past
  FP32_ORDER's 1e-6 floor (``attn_v_bcast_48``: 1.3e-5), where the mirror
  holds it exactly;
* poison: P is filled with 1e3 before the score steps, so a row a producer
  fails to write survives as poison;
* a mutation: one head's block rolled by a row, written over P in place, must
  move that head's output, so the checks above would catch a shifted block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ipu_harness.chain import Chain, Resident
from ipu_harness.packing import stored_rows
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat
from ipu_isa.geometry import LANES
from ipu_kernels.reference.datapath import acc_fold, agg_fold

FP32 = ElementFormat.FP32
POISON = 1e3


@dataclass(frozen=True)
class Seam:
    """A score kernel feeding an attn@V kernel, one score step per head.

    Attributes:
        scores: The producer (``qk_scores_*``: query-major P;
            ``attn_scores_km_*``: key-major P).
        values: The consumer (``attn_v_*`` or ``attn_v_bcast_*``).
        n_tok: Tokens.
        d: Head dimension.
        heads: Heads, one producer step each.
        seed: The old test's RandomState seed.
        tol: The old test's ``(rtol, atol)`` against the float64 two-stage
            reference.
    """

    scores: str
    values: str
    n_tok: int
    d: int
    heads: int
    seed: int
    tol: tuple[float, float]

    @property
    def key_major(self) -> bool:
        return self.scores.startswith("attn_scores_km")

    @property
    def row_chunks(self) -> int:
        """Rows per P row (a query's keys, or a key's queries)."""
        return -(-self.n_tok // LANES)

    @property
    def head_rows(self) -> int:
        """The rows of one head's P block: one producer step's output."""
        return self.n_tok * self.row_chunks


SEAMS = {
    "qk_scores_16x60-attn_v_16x60": Seam(
        "qk_scores_16x60", "attn_v_16x60", 16, 60, 4, 0x51E0, (1e-4, 1e-3)
    ),
    "qk_scores_256x36-attn_v_256x36": Seam(
        "qk_scores_256x36", "attn_v_256x36", 256, 36, 4, 0x5EA3, (1e-3, 1e-2)
    ),
    "attn_scores_km_16x60-attn_v_bcast_60": Seam(
        "attn_scores_km_16x60", "attn_v_bcast_60", 16, 60, 4, 0x5B10, (1e-4, 1e-3)
    ),
    "attn_scores_km_256x36-attn_v_bcast_36": Seam(
        "attn_scores_km_256x36", "attn_v_bcast_36", 256, 36, 4, 0x6EA3, (1e-3, 1e-2)
    ),
}


def _qkv(seam: Seam, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, K ``[heads, d, n_tok]`` channel-major and V ``[heads, d, n_tok]``, uniform(-1, 1)."""
    rng = np.random.RandomState(seed)
    shape = (seam.heads, seam.d, seam.n_tok)
    return tuple(rng.uniform(-1.0, 1.0, size=shape).astype(np.float32) for _ in range(3))


def _producer_inputs(seam: Seam, q: np.ndarray, k: np.ndarray) -> dict[str, np.ndarray]:
    """One head's score inputs: ``y = a @ b`` is ``Q^T K`` (query-major) or ``K^T Q`` (key-major)."""
    if seam.key_major:
        return {"b": q, "a": np.ascontiguousarray(k.T)}
    return {"b": k, "a": np.ascontiguousarray(q.T)}


def _p_rows(chain: Chain, seam: Seam) -> Resident:
    """P, reserved and poisoned, above the rows the producer pins its ``b`` to.

    ``qk_scores_16x60``, ``qk_scores_256x36`` and the ``attn_scores_km``
    kernels of the same layers read ``b`` through CR0 (hardwired 0), so ``b``
    is pinned at row 0; a guard of those rows keeps P off them.
    """
    manifest = default_registry().kernel(seam.scores)
    plan = plan_case(manifest, {"n_tok": seam.n_tok, "d": seam.d}, FP32)
    guard = chain.reserve(stored_rows(manifest.tensors["b"], plan))
    p = chain.reserve(seam.heads * seam.head_rows, pack="padded")
    chain.release(guard)
    chain.write(np.full((p.rows, LANES), POISON), at=p)
    return p


def _score_into(chain: Chain, seam: Seam, p: Resident, q: np.ndarray, k: np.ndarray) -> None:
    """Each head's score step, its output written into head h's block of P."""
    params = {"n_tok": seam.n_tok, "d": seam.d}
    for h in range(seam.heads):
        step = chain.run(
            seam.scores,
            params,
            _producer_inputs(seam, q[h], k[h]),
            check=True,
            at={"y": p.slice(h * seam.head_rows, seam.head_rows)},
        )
        assert step.result.passed, step.result.describe()
        assert step.resident["y"].rows == seam.head_rows  # one whole row per key or query


def _p_values(chain: Chain, seam: Seam, p: Resident) -> np.ndarray:
    """P as the consumer reads it, query-major ``[heads, i, s]``: the live lanes."""
    rows = chain.read(p).reshape(seam.heads, seam.n_tok, seam.row_chunks * LANES)
    live = rows[..., : seam.n_tok].astype(np.float32)
    return np.swapaxes(live, -1, -2) if seam.key_major else live


def _attend(chain: Chain, seam: Seam, p: Resident, v: np.ndarray) -> np.ndarray:
    """The attn@V step on P where it lies; its output ``[heads, d, n_tok]``."""
    step = chain.run(seam.values, {"n_tok": seam.n_tok, "d": seam.d}, {"b": p, "a": v})
    assert step.result.passed, step.result.describe()  # halted, no fault
    return np.asarray(step.result.outputs["y"], dtype=np.float32)


def _two_stage(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``O[h, t, i] = sum_s (sum_c Q[h, c, i] K[h, c, s]) V[h, t, s]`` in float64.

    Raw scores, no softmax: what the scores kernel emits is what attn@V takes
    as P."""
    s = np.einsum("hci,hcs->his", q.astype(np.float64), k.astype(np.float64))
    return np.einsum("his,hts->hti", s, v.astype(np.float64))


@pytest.mark.parametrize("seam", SEAMS.values(), ids=SEAMS.keys())
def test_scores_feed_attn_v_in_place(seam: Seam):
    q, k, v = _qkv(seam, seam.seed)
    chain = Chain(FP32)
    p = chain.hold(_p_rows(chain, seam))
    _score_into(chain, seam, p, q, k)
    p_qm = _p_values(chain, seam, p)
    assert not np.any(p_qm == np.float32(POISON)), "a producer left poison in P's live lanes"

    got = _attend(chain, seam, p, v)
    rtol, atol = seam.tol
    np.testing.assert_allclose(got, _two_stage(q, k, v), rtol=rtol, atol=atol)
    mirror = acc_fold(p_qm, v) if seam.key_major else agg_fold(p_qm, v)
    np.testing.assert_array_equal(got, mirror)

    # One head's block rolled by a row, as a producer with the wrong base or
    # pitch would leave it: that head's output must move.
    h = 1
    rows = chain.read(p)
    lo, hi = h * seam.head_rows, (h + 1) * seam.head_rows
    rows[lo:hi] = np.roll(rows[lo:hi], -1, axis=0)
    chain.write(rows, at=p)
    mutated = _attend(chain, seam, p, v)
    assert not np.allclose(mutated[h], got[h], rtol=rtol, atol=atol), (
        "a block rolled by a row did not move its head's output: the seam check has no teeth"
    )
    np.testing.assert_array_equal(np.delete(mutated, h, axis=0), np.delete(got, h, axis=0))


@pytest.mark.parametrize(
    "seam", [s for s in SEAMS.values() if s.n_tok == 16], ids=lambda s: s.scores
)
def test_each_head_lands_in_its_own_block(seam: Seam):
    """Head h's Q is h + 1 and K is 1, so every score of head h is d * (h + 1):
    the block at head slot h holds head h's scores, in its live lanes."""
    ones = np.ones((seam.heads, seam.d, seam.n_tok), np.float32)
    q = ones * np.arange(1, seam.heads + 1, dtype=np.float32)[:, None, None]
    chain = Chain(FP32)
    p = chain.hold(_p_rows(chain, seam))
    _score_into(chain, seam, p, q, ones)
    p_qm = _p_values(chain, seam, p)
    for h in range(seam.heads):
        np.testing.assert_array_equal(p_qm[h], np.float32(seam.d * (h + 1)))


# -- attn_scores_km_64x48 -> attn_v_bcast_48 (L4) ------------------------------
#
# attn_scores_km_64x48 scores one head of all 4 streams per run: blocks (p, h)
# for p = 0..3. attn_v_bcast_48's P holds the 16 (stream, head) blocks at
# b = p * 4 + h, so a producer's four blocks are not contiguous there: the host
# moves each block to its slot, rows unchanged.

KM64, BCAST48 = "attn_scores_km_64x48", "attn_v_bcast_48"
STREAMS, HEADS48, N48, D48 = 4, 4, 64, 48


def _km64_chain(seed: int) -> tuple[Chain, Resident, np.ndarray, np.ndarray]:
    """The four producer steps, their blocks moved into P. Returns the chain,
    P, the float64 two-stage reference ``[16, d, n]``, and V."""
    rng = np.random.RandomState(seed)
    shape = (STREAMS, HEADS48, D48, N48)
    q = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    k = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, size=(STREAMS * HEADS48, D48, N48)).astype(np.float32)
    chain = Chain(FP32)
    p = chain.hold(chain.reserve(STREAMS * HEADS48 * N48, pack="padded"))
    chain.write(np.full((p.rows, LANES), POISON), at=p)
    for h in range(HEADS48):
        step = chain.run(
            KM64,
            {"n_tok": N48, "d": D48, "head": h},
            {"b": q[:, h], "a": np.ascontiguousarray(np.swapaxes(k[:, h], -1, -2))},
            check=True,
        )
        assert step.result.passed, step.result.describe()
        blocks = chain.read(step.resident["y"]).reshape(STREAMS, N48, LANES)
        for stream in range(STREAMS):
            chain.write(blocks[stream], at=p.slice((stream * HEADS48 + h) * N48, N48))
    # S[p, h][key, query] = sum_c Q[p, h, c, query] K[p, h, c, key]
    s = np.einsum("phcq,phck->phkq", q.astype(np.float64), k.astype(np.float64))
    want = np.einsum("bkq,btk->btq", s.reshape(-1, N48, N48), v.astype(np.float64))
    return chain, p, want, v


def _bcast48(chain: Chain, p: Resident, v: np.ndarray) -> np.ndarray:
    step = chain.run(BCAST48, {"n_tok": N48, "d": D48}, {"b": p, "a": v})
    assert step.result.passed, step.result.describe()  # halted, no fault
    return np.asarray(step.result.outputs["y"], dtype=np.float32)


def test_km64_scores_feed_attn_v_bcast_48():
    chain, p, want, v = _km64_chain(0xC48)
    p_rows = chain.read(p).reshape(-1, N48, LANES)[..., :N48]
    assert not np.any(p_rows == POISON), "a block of P was never written"
    got = _bcast48(chain, p, v)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-2)
    np.testing.assert_array_equal(got, acc_fold(np.swapaxes(p_rows, -1, -2), v))


def test_km64_scores_at_the_wrong_pitch_are_caught():
    """The producer's rows packed at a 64-element key pitch (cropped to the
    live lanes) instead of a whole row per key: the output moves."""
    chain, p, _, v = _km64_chain(0xC49)
    correct = _bcast48(chain, p, v)
    rows = chain.read(p)
    cropped = rows.reshape(-1, LANES)[:, :N48].reshape(-1, LANES)
    mispitched = np.zeros_like(rows)
    mispitched[: cropped.shape[0]] = cropped
    chain.write(mispitched, at=p)
    diff = np.abs(_bcast48(chain, p, v) - correct)
    assert float(diff.max()) > 1e-2, f"a mispitched P moved the output by only {diff.max():.3e}"
