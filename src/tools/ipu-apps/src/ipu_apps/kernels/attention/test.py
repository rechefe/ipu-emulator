"""Attention family tests, beyond each kernel's own cases and test.py.

* **Chain separation** -- the query-major (``qk_scores`` -> ``attn_v``, AGG)
  and key-major (``attn_scores_km`` -> ``attn_v_bcast``, ACC.ADD) chains are
  bit-different by design and must never be mixed. Every kernel's ``op``
  names its chain, so the registry can never hand one chain's consumer to the
  other chain's scores.
* **Seams** -- run two REAL kernel binaries back to back and feed the
  producer's raw XMEM store output, byte for byte, as the consumer's staged
  input file, with no reshape/crop in between (per-kernel tests hand-stage
  numpy arrays into each harness's layout and cannot see a pitch, base, extent
  or ordering mismatch between them). Each seam is checked against a numpy
  reference of the WHOLE two-stage computation, and each has a mutation check
  proving the assertion would catch a shifted block.
* **Softmax feed** -- the L5 producers' uncropped output against the real
  softmax kernels: verbatim it is refused, cropped to the live lanes it
  computes the right softmax.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import pytest

from ipu_apps.kernel_registry import kernels, resolve
from ipu_apps.kernel_registry.cases import assemble_kernel
from ipu_apps.kernels.attention.app import CHAIN_OF, CHAINS
from ipu_apps.kernels.attention.cases import acc_fold, agg_fold
from ipu_apps.kernels.attention.attn_scores_km_16x60 import app as km16
from ipu_apps.kernels.attention.attn_scores_km_256x36 import app as km256
from ipu_apps.kernels.attention.attn_scores_km_64x48 import app as km64
from ipu_apps.kernels.attention.attn_v_16x60 import app as av16
from ipu_apps.kernels.attention.attn_v_256x36 import app as av256
from ipu_apps.kernels.attention.attn_v_bcast_36 import app as bc36
from ipu_apps.kernels.attention.attn_v_bcast_48 import app as bc48
from ipu_apps.kernels.attention.attn_v_bcast_60 import app as bc60
from ipu_apps.kernels.attention.qk_scores_16x60 import app as qk16
from ipu_apps.kernels.attention.qk_scores_256x36 import app as qk256

MAX_CYCLES = 20_000_000
LANES = 128
ROW_BYTES = LANES * 4


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    """``inst(kernel)``: the kernel's assembled binary, assembled once per module."""
    root = tmp_path_factory.mktemp("inst")
    cache = {}

    def get(kernel):
        if kernel not in cache:
            cache[kernel] = assemble_kernel(kernel, root)
        return cache[kernel]

    return get


def _run(app, *, poison=()):
    """Run ``app`` on fresh FP32 state, first filling each ``(address, rows)``
    region in ``poison`` with 1e3 so a row the kernel fails to write survives
    as poison. Returns the halted state."""
    state = app.make_state()
    for address, rows in poison:
        state.xmem.write_address(address, bytearray(np.full(rows * LANES, POISON, "<f4").tobytes()))
    state, cycles = app.run(max_cycles=MAX_CYCLES, state=state)
    assert cycles > 0 and state.is_halted
    return state


POISON = np.float32(1e3)


# -- chain separation ---------------------------------------------------------

def _attention_kernels():
    return [s for s in kernels() if s.resource_package.startswith("ipu_apps.kernels.attention.")]


def test_every_kernel_belongs_to_exactly_one_chain():
    specs = _attention_kernels()
    assert len(specs) == 12
    for spec in specs:
        assert spec.op in CHAIN_OF, f"{spec.name}: op {spec.op!r} is in no chain"
        chain = CHAIN_OF[spec.op]
        other = next(c for c in CHAINS if c != chain)
        assert chain in spec.tags and other not in spec.tags, (spec.name, spec.tags)


def test_chains_never_route_to_each_other():
    """Whatever a query carries, an op only ever resolves to its own chain.

    Both attn@V kernels of a layer accept the same ``(n_tok, d)``, so only the
    op can keep a caller holding key-major scores away from the AGG kernel
    (whose P input is query-major) and vice versa.
    """
    params = [dict(n_tok=n, d=d) for n, d in ((16, 60), (64, 48), (256, 36))]
    for op, chain in CHAIN_OF.items():
        for query in params:
            verdict = resolve(op, **query)
            assert verdict, f"{op} {query}: {verdict.reason}"
            assert verdict.kernel.op == op and chain in verdict.kernel.tags
            assert not verdict.alternatives, verdict.alternatives


def test_attn_v_bcast_n_tok_must_match_when_given():
    assert resolve("attn_v_bcast", d=36).app_name == "attn_v_bcast_36"
    verdict = resolve("attn_v_bcast", n_tok=16, d=36)
    assert not verdict and "n_tok=256" in verdict.reason


def test_attn_scores_km_head_reaches_the_constructor():
    verdict = resolve("attn_scores_km", n_tok=16, d=60, head=3)
    assert verdict and verdict.kwargs == {"head": 3}
    assert resolve("attn_scores_km", n_tok=16, d=60).kwargs == {"head": 0}
    refused = resolve("attn_scores_km", n_tok=16, d=60, head=4)
    assert not refused and "head" in refused.reason


# -- seam: qk_scores_256x36 -> attn_v_256x36 (query-major + AGG, L3) ----------
#
# qk_scores_256x36 has no internal head slicing: it consumes a single D=36
# channel-major Q/K pair. Run it once per head and concatenate the four raw
# output files in head order as attn_v_256x36's P.
#
#   qk_scores_256x36 per-head output: N * N_TG = 512 rows, row (i, g) at
#     S_BASE_ROW + i*N_TG + g  (query-major, group-interleaved).
#   attn_v_256x36 P input: P[i, s] at PBASE + h*P_HEAD_STRIDE_ROWS(=512)
#     + i*PV_STRIDE_ROWS(=2) + s//128 rows (4 heads, head-major).
#
# P_HEAD_STRIDE_ROWS equals qk_scores's single-head output row count and the
# intra-block formula (i*2 + g) is identical on both sides, so the
# concatenation *should* line up -- this verifies it empirically.

assert qk256.N == av256.N_TOK and qk256.D * av256.N_HEAD == av256.N_CHAN


def _qk256_raw(inst, tmp_path, q_head, k_head, tag):
    """Run qk_scores_256x36 for one head's [D, N] Q/K; return its raw output bytes."""
    q_path, k_path = tmp_path / f"q_{tag}.bin", tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(q_head.astype(np.float32).tobytes())
    k_path.write_bytes(k_head.astype(np.float32).tobytes())
    out = tmp_path / f"qk_out_{tag}.bin"
    _run(qk256.QkScores256x36App(inst_path=inst("qk_scores_256x36"), query_path=q_path,
                                 key_path=k_path, output_path=out),
         poison=[(qk256.S_BASE, qk256.N * qk256.N_TG)])
    raw = out.read_bytes()
    assert len(raw) == qk256.N * qk256.N_TG * qk256.OUTPUT_ROW_BYTES
    # Poison detector: no output element should still read as the poison
    # value (extremely unlikely from real uniform(-1,1) products/sums).
    assert not np.any(np.frombuffer(raw, np.float32) == POISON), (
        f"{tag}: producer under-wrote its output region -- poison survived")
    return raw


def _v_rows(V, width):
    """V[h, t, s] channel-major: one row of ``width`` lanes per value channel."""
    return np.ascontiguousarray(
        np.pad(V.reshape(-1, V.shape[-1]), ((0, 0), (0, width - V.shape[-1])))).astype(np.float32)


def _attn_v256(inst, tmp_path, p_bytes, V, tag):
    p_path, v_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("p", "v", "o"))
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(_v_rows(V, av256.PV_STRIDE_ROWS * LANES).tobytes())
    _run(av256.AttnV256x36App(inst_path=inst("attn_v_256x36"), p_path=p_path, v_path=v_path,
                              output_path=out),
         poison=[(av256.OBASE, av256.O_ROWS)])
    raw = np.fromfile(out, dtype="<f4")
    assert raw.size == av256.N_CHAN * 2 * LANES
    assert not np.any(raw == POISON), "attn_v_256x36 under-wrote its output region"
    return raw.reshape(av256.N_HEAD, av256.D, 2 * LANES)[..., :av256.N_TOK]


def _two_stage(Q, K, V):
    """O[h, t, i] = sum_s (sum_c Q[h,c,i] K[h,c,s]) V[h,t,s], in float64.

    Raw scores, no softmax: the chain under test is "whatever the scores
    kernel emits is what attn@V consumes as P", regardless of whether a
    softmax would normally sit in between.
    """
    S = np.einsum("hci,hcs->his", Q.astype(np.float64), K.astype(np.float64))
    return np.einsum("his,hts->hti", S, V.astype(np.float64))


def _qkv(seed, heads, d, n):
    rng = np.random.RandomState(seed)
    return [rng.uniform(-1.0, 1.0, size=(heads, d, n)).astype(np.float32) for _ in range(3)]


def test_seam_qk_scores_to_attn_v_256x36(inst, tmp_path):
    Q, K, V = _qkv(0x5EA3, av256.N_HEAD, qk256.D, qk256.N)
    p_bytes = b"".join(_qk256_raw(inst, tmp_path, Q[h], K[h], f"h{h}") for h in range(av256.N_HEAD))
    assert len(p_bytes) == av256.N_HEAD * av256.P_HEAD_STRIDE_ROWS * ROW_BYTES, (
        "concatenated qk_scores output size does not equal attn_v_256x36's "
        "expected P region size -- head-block row counts disagree")
    got = _attn_v256(inst, tmp_path, p_bytes, V, "chained")
    np.testing.assert_allclose(got, _two_stage(Q, K, V), rtol=1e-3, atol=1e-2,
                               err_msg="chained qk_scores->attn_v mismatch")


def test_seam_qk_attn_v_head_block_pitch_mutation_detected(inst, tmp_path):
    """Harness-teeth check: roll one head's block by one row -- a producer
    that shifted its row pitch/base, analogous to the attn_scores_km_64x48 /
    attn_v_bcast_48 sub-row-pitch bug -- run it through the REAL
    attn_v_256x36, and confirm the seam assertion fails for that head."""
    Q, K, V = _qkv(0x5EA4, av256.N_HEAD, qk256.D, qk256.N)
    good = b"".join(_qk256_raw(inst, tmp_path, Q[h], K[h], f"mut{h}") for h in range(av256.N_HEAD))
    h_mut = 2
    block = av256.P_HEAD_STRIDE_ROWS * ROW_BYTES
    lo, hi = h_mut * block, (h_mut + 1) * block
    mutated = good[:lo] + good[lo + ROW_BYTES:hi] + good[lo:lo + ROW_BYTES] + good[hi:]
    assert mutated != good, "mutation harness produced no actual change"
    got = _attn_v256(inst, tmp_path, mutated, V, "mut")
    expected = _two_stage(Q, K, V)
    assert not np.allclose(got[h_mut], expected[h_mut], rtol=1e-3, atol=1e-2), (
        "mutated (row-rolled) head block did not produce any numeric "
        "mismatch -- the seam assertion would not have caught this bug")


# -- seam: attn_scores_km_256x36 -> attn_v_bcast_36 (key-major, L3) -----------
#
# attn_scores_km_256x36 takes `head` and slices a canonical [N_HEADS*D, N_TOK]
# Q/K file itself. Its per-head output is N_TOK * N_TG = 512 rows, row (s, g)
# at SBASE_ROW + s*N_TG + g (key-major: lane = query). attn_v_bcast_36's P is
# P[i, s] at PBASE + h*512 + s*PV_STRIDE_ROWS(=2) + i//128 rows; PV_STRIDE_ROWS
# == N_TG == 2, so the intra-block row formula is identical.

assert km256.N_HEADS == bc36.N_HEAD and bc36.PV_STRIDE_ROWS == km256.N_TG


def _km256_raw(inst, q_path, k_path, head, out):
    _run(km256.AttnScoresKM256x36App(inst_path=inst("attn_scores_km_256x36"), input_path=q_path,
                                     weights_path=k_path, output_path=out, head=head),
         poison=[(km256.SBASE, km256.N_TOK * km256.N_TG)])
    raw = out.read_bytes()
    assert len(raw) == km256.N_TOK * km256.N_TG * km256.OUTPUT_ROW_BYTES
    assert not np.any(np.frombuffer(raw, np.float32) == POISON), (
        f"head {head}: producer under-wrote its output region -- poison survived")
    return raw


def _km256_chain(inst, tmp_path, seed, tag):
    rng = np.random.RandomState(seed)
    n_chan = km256.N_HEADS * km256.D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, km256.N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, km256.N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(bc36.N_HEAD, bc36.D, bc36.N_TOK)).astype(np.float32)
    q_path, k_path = tmp_path / f"q_{tag}.bin", tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    p_bytes = b"".join(_km256_raw(inst, q_path, k_path, h, tmp_path / f"km_{tag}{h}.bin")
                       for h in range(km256.N_HEADS))
    assert len(p_bytes) == bc36.N_HEAD * bc36.P_HEAD_STRIDE_ROWS * ROW_BYTES, (
        "concatenated attn_scores_km output size does not equal attn_v_bcast_36's "
        "expected P region size -- head-block row counts disagree")
    shape = (km256.N_HEADS, km256.D, km256.N_TOK)
    return p_bytes, _two_stage(Q.reshape(shape), K.reshape(shape), V), V


def _bcast36(inst, tmp_path, p_bytes, V, tag):
    p_path, v_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("p", "v", "o"))
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(_v_rows(V, bc36.PV_STRIDE_ROWS * LANES).tobytes())
    _run(bc36.AttnVBcast36App(inst_path=inst("attn_v_bcast_36"), p_path=p_path, v_path=v_path,
                              output_path=out),
         poison=[(bc36.OBASE, bc36.O_ROWS)])
    raw = np.fromfile(out, dtype="<f4")
    assert raw.size == bc36.N_CHAN * 2 * LANES
    assert not np.any(raw == POISON), "attn_v_bcast_36 under-wrote its output region"
    return raw.reshape(bc36.N_HEAD, bc36.D, 2 * LANES)[..., :bc36.N_TOK]


def test_seam_attn_scores_km_to_attn_v_bcast_36(inst, tmp_path):
    p_bytes, expected, V = _km256_chain(inst, tmp_path, 0x6EA3, "ok")
    got = _bcast36(inst, tmp_path, p_bytes, V, "chained")
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-2,
                               err_msg="chained attn_scores_km->attn_v_bcast mismatch")


def test_seam_km_bcast_head_block_pitch_mutation_detected(inst, tmp_path):
    """Harness-teeth check: roll head 1's block by one row -- the exact class
    of bug this seam exists to find (attn_scores_km_64x48 sub-row pitch vs
    attn_v_bcast_48's whole-row read) -- and confirm the assertion fails."""
    good, expected, V = _km256_chain(inst, tmp_path, 0x6EA4, "mut")
    h_mut = 1
    block = bc36.P_HEAD_STRIDE_ROWS * ROW_BYTES
    lo, hi = h_mut * block, (h_mut + 1) * block
    mutated = good[:lo] + good[lo + ROW_BYTES:hi] + good[lo:lo + ROW_BYTES] + good[hi:]
    assert mutated != good, "mutation harness produced no actual change"
    got = _bcast36(inst, tmp_path, mutated, V, "mut")
    assert not np.allclose(got[h_mut], expected[h_mut], rtol=1e-3, atol=1e-2), (
        "mutated (row-rolled) head block did not produce any numeric "
        "mismatch -- the seam assertion would not have caught this bug")


# -- seam: qk_scores_16x60 -> attn_v_16x60 (query-major + AGG, L5) ------------
#
#   qk_scores_16x60 -> S[i, s] query-major, one head per run, one row per
#                      query i (lane s = key s).
#   attn_v_16x60    -> expects P query-major: P[h, i, :N_TOK] = "row i = all
#                      keys for query i", heads concatenated at
#                      PBASE + h*P_HEAD_STRIDE.
#
# qk_scores_16x60 computes ONE head per invocation, so it runs 4 times and the
# 4 raw output files are concatenated in head order -- exactly the block index
# attn_v_16x60 expects.

assert qk16.N == av16.N_TOK and qk16.D == av16.D and qk16.N_TG == 1, (
    "qk_scores_16x60 / attn_v_16x60 shape constants diverged")


def _qk16_raw(inst, tmp_path, q_head, k_head, tag):
    """Run the REAL qk_scores_16x60 for one head; return its raw output bytes."""
    q_path, k_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("q", "k", "s"))
    q_path.write_bytes(q_head.tobytes())
    k_path.write_bytes(k_head.tobytes())
    _run(qk16.QkScores16x60App(inst_path=inst("qk_scores_16x60"), query_path=q_path,
                               key_path=k_path, output_path=out),
         poison=[(qk16.S_BASE, qk16.N * qk16.N_TG)])
    raw = out.read_bytes()
    assert not np.any(np.frombuffer(raw, np.float32) == POISON), (
        f"{tag}: producer under-wrote its output region -- poison survived")
    return raw


def _qk16_chain(inst, tmp_path, Q, K):
    """qk_scores_16x60 once per head, raw outputs concatenated in head order --
    exactly the bytes attn_v_16x60.setup() writes verbatim at PBASE. Q, K are
    [N_HEAD*D, N_TOK] canonical channel-major."""
    D = av16.D
    parts = []
    for h in range(av16.N_HEAD):
        raw = _qk16_raw(inst, tmp_path, Q[h * D:(h + 1) * D], K[h * D:(h + 1) * D], f"h{h}")
        assert len(raw) == av16.P_HEAD_STRIDE_ROWS * ROW_BYTES, (
            f"head {h}: qk_scores_16x60 produced {len(raw)} B, expected "
            f"{av16.P_HEAD_STRIDE_ROWS * ROW_BYTES} B (P_HEAD_STRIDE)")
        parts.append(raw)
    return b"".join(parts)


def _attn_v16(inst, tmp_path, p_bytes, V, tag):
    """Run the REAL attn_v_16x60 on a raw P blob, its O region poisoned first."""
    p_path, v_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("p", "v", "o"))
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(_v_rows(V, av16.PV_STRIDE_ROWS * LANES).tobytes())
    _run(av16.AttnV16x60App(inst_path=inst("attn_v_16x60"), p_path=p_path, v_path=v_path,
                            output_path=out),
         poison=[(av16.OBASE, av16.O_ROWS)])
    got = np.fromfile(out, dtype="<f4").reshape(av16.N_HEAD, av16.D, LANES)[..., :av16.N_TOK]
    assert not np.any(got == POISON), (
        f"{tag}: attn_v_16x60 under-wrote its output region -- poison survived "
        f"into the valid-lane bytes")
    return got


def _scores16(Q, K, heads, d):
    """S[h, i, s] = sum_c Q[h*d + c, i] K[h*d + c, s] as the float32 matmul."""
    return np.stack([Q[h * d:(h + 1) * d].T @ K[h * d:(h + 1) * d] for h in range(heads)])


def _qkv16(seed):
    rng = np.random.RandomState(seed)
    n_chan = av16.N_HEAD * av16.D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, av16.N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, av16.N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(av16.N_HEAD, av16.D, av16.N_TOK)).astype(np.float32)
    return Q, K, V


def test_seam_qk_to_attn_v_16x60_agrees(inst, tmp_path):
    """Full chain against a numpy reference of the WHOLE two-stage computation
    (QK^T, then attn_v_16x60's own AGG-mirroring fold), not each kernel's
    isolated golden."""
    Q, K, V = _qkv16(0x51E0)
    p_bytes = _qk16_chain(inst, tmp_path, Q, K)
    assert len(p_bytes) == av16.P_ROWS * ROW_BYTES, (
        f"chained P is {len(p_bytes)} B, attn_v_16x60 expects {av16.P_ROWS * ROW_BYTES} B "
        f"-- producer/consumer region-size mismatch")
    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(av16.P_ROWS, LANES)
    assert not np.any(p_arr[:, :av16.N_TOK] == POISON), (
        "producer failed to overwrite poison in its declared-valid lanes")
    got = _attn_v16(inst, tmp_path, p_bytes, V, "chained")
    expected = agg_fold(_scores16(Q, K, av16.N_HEAD, av16.D), V)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3, err_msg=(
        "chained qk_scores_16x60 -> attn_v_16x60 mismatch against the full "
        "two-stage numpy reference"))


def test_seam_qk_to_attn_v_16x60_head_concat_order(inst, tmp_path):
    """Head-identifiable data (head h's Q = h+1, K = 1, so S = D*(h+1)) shows
    that concatenating the single-head runs in HEAD ORDER lands each head's
    block where attn_v_16x60 indexes it (PBASE + h*P_HEAD_STRIDE)."""
    H, D, N = av16.N_HEAD, av16.D, av16.N_TOK
    Q = np.repeat(np.arange(1, H + 1, dtype=np.float32), D)[:, None] * np.ones((1, N), np.float32)
    K = np.ones((H * D, N), dtype=np.float32)
    p_arr = np.frombuffer(_qk16_chain(inst, tmp_path, Q, K), dtype=np.float32)
    p_arr = p_arr.reshape(H, av16.P_HEAD_STRIDE_ROWS, LANES)
    for h in range(H):
        np.testing.assert_allclose(p_arr[h, :, :N], np.float32(D * (h + 1)), rtol=1e-5, err_msg=(
            f"block at head-slot {h} does not contain head {h}'s scores -- head "
            f"concatenation order does not match attn_v_16x60's PBASE + h*P_HEAD_STRIDE"))


def test_seam_qk_to_attn_v_16x60_mutation_kills_test(inst, tmp_path):
    """Harness-teeth check: shift the chained P blob by one row (drop the
    first, pad a garbage row) -- the pitch/base mismatch class documented for
    attn_scores_km_64x48/attn_v_bcast_48 -- and confirm the comparison
    against the correctly-aligned reference FAILS."""
    Q, K, V = _qkv16(0x51E1)
    p_bytes = _qk16_chain(inst, tmp_path, Q, K)
    mutated = p_bytes[ROW_BYTES:] + np.full(LANES, POISON, dtype=np.float32).tobytes()
    assert len(mutated) == len(p_bytes)
    got = _attn_v16(inst, tmp_path, mutated, V, "mutated")
    expected = agg_fold(_scores16(Q, K, av16.N_HEAD, av16.D), V)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


def test_seam_qk_to_attn_v_16x60_shared_state_zero_copy(inst, tmp_path):
    """The same chain, but both kernels run against ONE shared IpuState with
    base rows placed so each head's qk_scores_16x60 run writes its S output
    DIRECTLY into attn_v_16x60's P slot for that head -- no file round-trip, no
    second state, no Python byte copy of the scores. Only Q/K (real inputs)
    and V (no producer kernel) are staged from disk.

    Made possible by constructor-parameterized base rows (qrow_base_row/
    s_base_row on QkScores16x60App; pbase_row/vbase_row/obase_row on
    AttnV16x60App) plus p_path=None/v_path=None on AttnV16x60App, which skips
    its own staging so the bytes already in the shared XMEM survive. These are
    placement concerns, not routed parameters, so the apps are constructed
    directly rather than through the registry.

    K is NOT relocatable: qk_scores_16x60.asm reads K_BASE from cr0, which the
    ISA hardwires to 0, so K always occupies rows [0, K_ROWS) and every other
    region here is placed above it.
    """
    Q, K, V = _qkv16(0x51E2)
    D = av16.D
    # Layout (rows), all in the ONE shared state:
    #   [0 .. K_ROWS)            K (fixed: cr0 is hardwired 0)
    #   [K_ROWS .. +P_ROWS)      P / S (qk_scores writes, attn_v reads, in place)
    #   [.. +V_ROWS)             V (host-staged; no producer)
    #   [.. +O_ROWS)             O (attn_v's output)
    #   [.. +QROW_ROWS)          qk_scores_16x60's staged-Q scratch (per head)
    p_base_row = qk16.K_ROWS
    v_base_row = p_base_row + av16.P_ROWS
    o_base_row = v_base_row + av16.V_ROWS
    qk_staging_base_row = o_base_row + av16.O_ROWS

    state = None
    for h in range(av16.N_HEAD):
        q_path, k_path = tmp_path / f"q_shared_h{h}.bin", tmp_path / f"k_shared_h{h}.bin"
        q_path.write_bytes(Q[h * D:(h + 1) * D].tobytes())
        k_path.write_bytes(K[h * D:(h + 1) * D].tobytes())
        qk_app = qk16.QkScores16x60App(
            inst_path=inst("qk_scores_16x60"), query_path=q_path, key_path=k_path,
            output_path=None,
            qrow_base_row=qk_staging_base_row,
            s_base_row=p_base_row + h * av16.P_HEAD_STRIDE_ROWS,  # attn_v's P slot for head h
        )
        if state is None:
            state = qk_app.make_state()
            state.xmem.write_address(v_base_row * ROW_BYTES, bytearray(
                _v_rows(V, av16.PV_STRIDE_ROWS * LANES).tobytes()))
        # run() reloads the program but never resets the program counter of a
        # state a prior run left halted.
        state.program_counter = 0
        _, cycles = qk_app.run(max_cycles=MAX_CYCLES, state=state)
        assert cycles > 0

    state.program_counter = 0
    av_app = av16.AttnV16x60App(
        inst_path=inst("attn_v_16x60"),
        p_path=None,   # P already in shared XMEM from the qk_scores runs above
        v_path=None,   # V already in shared XMEM, staged above
        output_path=None,
        pbase_row=p_base_row, vbase_row=v_base_row, obase_row=o_base_row,
    )
    _, cycles = av_app.run(max_cycles=MAX_CYCLES, state=state)
    assert cycles > 0

    raw = state.xmem.read_address(o_base_row * ROW_BYTES, av16.N_CHAN * av16.O_CHAN_BYTES)
    got = np.frombuffer(bytes(raw), dtype=np.float32).reshape(av16.N_HEAD, D, LANES)
    S = np.stack([Q[h * D:(h + 1) * D].T.astype(np.float64) @ K[h * D:(h + 1) * D].astype(np.float64)
                  for h in range(av16.N_HEAD)])
    expected = np.einsum("his,hts->hti", S, V.astype(np.float64))
    np.testing.assert_allclose(got[..., :av16.N_TOK].astype(np.float64), expected,
                               rtol=1e-4, atol=1e-3,
                               err_msg="shared-state chain mismatch vs numpy float64 reference")


# -- seam: attn_scores_km_16x60 -> attn_v_bcast_60 (key-major, L5) ------------
#
# The same kernel-shape class (N <= LANES, single token group, key-major
# restage) that produced the documented attn_scores_km_64x48/attn_v_bcast_48
# row-pitch bug: attn_scores_km_64x48
# used to crop its key rows to N*ELEM_BYTES before storing them, while
# attn_v_bcast_48 addressed one key per WHOLE 512 B row. Both L5 kernels use
# the FULL-ROW convention; this confirms it for the real store/load path.
#
#   attn_scores_km_16x60 -> S key-major, one head per run, one row per key s
#                           (lane i = query i).
#   attn_v_bcast_60      -> P key-major: row s = all queries for key s, heads
#                           concatenated at PBASE + h*P_HEAD_STRIDE.

assert (km16.N_TOK == bc60.N_TOK and km16.D == bc60.D and km16.N_TG == 1
        and km16.N_HEADS == bc60.N_HEAD), "attn_scores_km_16x60 / attn_v_bcast_60 diverged"


def _km16_raw(inst, tmp_path, Q, K, head, tag):
    """Run the REAL attn_scores_km_16x60 for one head of the FULL 4-head Q/K
    (it slices `head`'s channel block itself); return its raw output bytes."""
    q_path, k_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("q", "k", "s"))
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    _run(km16.AttnScoresKM16x60App(inst_path=inst("attn_scores_km_16x60"), input_path=q_path,
                                   weights_path=k_path, output_path=out, head=head),
         poison=[(km16.SBASE, km16.N_TOK * km16.N_TG)])
    raw = out.read_bytes()
    assert not np.any(np.frombuffer(raw, np.float32) == POISON), (
        f"{tag}: producer under-wrote its output region -- poison survived")
    return raw


def _km16_chain(inst, tmp_path, Q, K):
    parts = []
    for h in range(bc60.N_HEAD):
        raw = _km16_raw(inst, tmp_path, Q, K, h, f"h{h}")
        assert len(raw) == bc60.P_HEAD_STRIDE_ROWS * ROW_BYTES, (
            f"head {h}: attn_scores_km_16x60 produced {len(raw)} B, expected "
            f"{bc60.P_HEAD_STRIDE_ROWS * ROW_BYTES} B (P_HEAD_STRIDE)")
        parts.append(raw)
    return b"".join(parts)


def _bcast60(inst, tmp_path, p_bytes, V, tag):
    p_path, v_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("p", "v", "o"))
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(_v_rows(V, bc60.PV_STRIDE_ROWS * LANES).tobytes())
    _run(bc60.AttnVBcast60App(inst_path=inst("attn_v_bcast_60"), p_path=p_path, v_path=v_path,
                              output_path=out),
         poison=[(bc60.OBASE, bc60.O_ROWS)])
    got = np.fromfile(out, dtype="<f4").reshape(bc60.N_HEAD, bc60.D, LANES)[..., :bc60.N_TOK]
    assert not np.any(got == POISON), (
        f"{tag}: attn_v_bcast_60 under-wrote its output region -- poison survived "
        f"into the valid-lane bytes")
    return got


def test_seam_km_to_bcast_60_agrees(inst, tmp_path):
    """Full chain against the WHOLE two-stage reference using the BROADCAST
    (ACC.ADD) fold -- never the AGG fold from the sibling chain."""
    rng = np.random.RandomState(0x5B10)
    n_chan = bc60.N_HEAD * bc60.D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, bc60.N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, bc60.N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(bc60.N_HEAD, bc60.D, bc60.N_TOK)).astype(np.float32)
    p_bytes = _km16_chain(inst, tmp_path, Q, K)
    assert len(p_bytes) == bc60.P_ROWS * ROW_BYTES, (
        f"chained P is {len(p_bytes)} B, attn_v_bcast_60 expects {bc60.P_ROWS * ROW_BYTES} B "
        f"-- producer/consumer region-size mismatch")
    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(bc60.P_ROWS, LANES)
    assert not np.any(p_arr[:, :bc60.N_TOK] == POISON), (
        "producer failed to overwrite poison in its declared-valid lanes")
    got = _bcast60(inst, tmp_path, p_bytes, V, "chained")
    expected = acc_fold(_scores16(Q, K, bc60.N_HEAD, bc60.D), V)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3, err_msg=(
        "chained attn_scores_km_16x60 -> attn_v_bcast_60 mismatch against the "
        "full two-stage numpy reference"))


def test_seam_km_to_bcast_60_row_pitch_and_head_concat(inst, tmp_path):
    """On RAW STORED BYTES: (1) each key's row is a WHOLE 512 B row, matching
    attn_v_bcast_60's one-key-per-row addressing -- the axis the
    attn_scores_km_64x48/attn_v_bcast_48 bug broke; (2) head-order
    concatenation lands head h's block at PBASE + h*P_HEAD_STRIDE (cr7).
    Head h's Q = h+1 and K = 1, so S = D*(h+1) identifies each block."""
    H, D, N = bc60.N_HEAD, bc60.D, bc60.N_TOK
    Q = np.repeat(np.arange(1, H + 1, dtype=np.float32), D)[:, None] * np.ones((1, N), np.float32)
    K = np.ones((H * D, N), dtype=np.float32)

    raw_h0 = _km16_raw(inst, tmp_path, Q, K, 0, "pitchcheck")
    assert len(raw_h0) == N * ROW_BYTES, (
        f"attn_scores_km_16x60 raw output is {len(raw_h0)} B for {N} keys -- expected "
        f"exactly {N * ROW_BYTES} B ({ROW_BYTES} B/row, full row pitch)")
    arr_h0 = np.frombuffer(raw_h0, dtype=np.float32).reshape(N, LANES)
    np.testing.assert_allclose(arr_h0[:, :N], float(D), rtol=1e-5)

    p_arr = np.frombuffer(_km16_chain(inst, tmp_path, Q, K), dtype=np.float32)
    p_arr = p_arr.reshape(H, bc60.P_HEAD_STRIDE_ROWS, LANES)
    for h in range(H):
        np.testing.assert_allclose(p_arr[h, :, :N], np.float32(D * (h + 1)), rtol=1e-5, err_msg=(
            f"block at head-slot {h} does not contain head {h}'s scores -- head "
            f"concatenation order does not match attn_v_bcast_60's "
            f"PBASE + h*P_HEAD_STRIDE indexing (cr7=P_HEAD_STRIDE_ROWS)"))


def test_seam_km_to_bcast_60_mutation_kills_test(inst, tmp_path):
    """Harness-teeth check: shift the chained P blob by one row and confirm
    the real-reference comparison FAILS."""
    rng = np.random.RandomState(0x5B11)
    n_chan = bc60.N_HEAD * bc60.D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, bc60.N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, bc60.N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(bc60.N_HEAD, bc60.D, bc60.N_TOK)).astype(np.float32)
    p_bytes = _km16_chain(inst, tmp_path, Q, K)
    mutated = p_bytes[ROW_BYTES:] + np.full(LANES, POISON, dtype=np.float32).tobytes()
    assert len(mutated) == len(p_bytes)
    got = _bcast60(inst, tmp_path, mutated, V, "mutated")
    expected = acc_fold(_scores16(Q, K, bc60.N_HEAD, bc60.D), V)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


# -- seam: attn_scores_km_64x48 -> attn_v_bcast_48 (key-major, L4) ------------
#
# attn_scores_km_64x48 scores ONE head across all P=4 streams per run (256 key
# rows). attn_v_bcast_48 wants all N_BLOCK=16 (stream, head) blocks in one P
# file addressed as row (b * N_TOK) with b = p * N_HEAD + h. So the chain
# needs N_HEAD=4 producer runs, each contributing 4 blocks (p=0..3 at fixed h)
# that are NOT contiguous in the consumer's b ordering.

assert (km64.N == bc48.N_TOK and km64.D == bc48.D and km64.P == 4 and km64.N_HEAD == 4
        and bc48.N_BLOCK == km64.P * km64.N_HEAD)


def _km64_raw(inst, tmp_path, Q, K, head):
    """Run the REAL attn_scores_km_64x48 for one head, its whole output region
    poisoned first. Returns the raw (uncropped, P*N rows of ROW_BYTES) bytes."""
    q_path, k_path, out = (tmp_path / f"{n}_{head}.bin" for n in ("q", "k", "km_out"))
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    _run(km64.AttnScoresKM64x48App(inst_path=inst("attn_scores_km_64x48"), input_path=q_path,
                                   weights_path=k_path, output_path=out, head=head),
         poison=[(km64.SBASE, km64.S_ROWS)])
    raw = out.read_bytes()
    assert len(raw) == km64.P * km64.N * km64.OUTPUT_ROW_BYTES
    # Every row must have been written by the kernel's own store path.
    rows = np.frombuffer(raw, dtype=np.float32).reshape(km64.P * km64.N, LANES)
    untouched = np.all(rows == POISON, axis=1)
    assert not untouched.any(), (
        f"producer left {untouched.sum()} whole rows as untouched poison "
        f"(row indices {np.nonzero(untouched)[0].tolist()})")
    return raw


def _km64_p_file(head_outputs):
    """attn_v_bcast_48's P from 4 single-head producer runs: run h emits blocks
    (p, h) for p=0..3, which land at block b = p*N_HEAD + h."""
    out = bytearray(bc48.N_BLOCK * bc48.N_TOK * ROW_BYTES)
    for h, raw in head_outputs.items():
        rows = np.frombuffer(raw, dtype=np.float32).reshape(km64.P, km64.N, LANES)
        for p in range(km64.P):
            lo = (p * km64.N_HEAD + h) * bc48.N_TOK * ROW_BYTES
            out[lo:lo + bc48.N_TOK * ROW_BYTES] = rows[p].tobytes()
    return bytes(out)


def _qkv64(seed):
    rng = np.random.RandomState(seed)
    Q = rng.uniform(-1.0, 1.0, size=(km64.P, km64.N_HEAD, km64.D, km64.N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(km64.P, km64.N_HEAD, km64.D, km64.N)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(bc48.N_BLOCK, bc48.D, bc48.N_TOK)).astype(np.float32)
    return Q, K, V


def _bcast48(inst, tmp_path, p_bytes, V, tag):
    p_path, v_path, out = (tmp_path / f"{n}_{tag}.bin" for n in ("p", "v", "o"))
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(_v_rows(V, LANES).tobytes())
    _run(bc48.AttnVBcast48App(inst_path=inst("attn_v_bcast_48"), p_path=p_path, v_path=v_path,
                              output_path=out))
    return np.fromfile(out, dtype="<f4").reshape(bc48.N_BLOCK, bc48.D, bc48.N_TOK)


def test_seam_km_bcast_48_agrees(inst, tmp_path):
    """Full chain: 4 attn_scores_km_64x48 runs (one per head) feed one
    attn_v_bcast_48 run, against a from-scratch two-stage numpy reference."""
    Q, K, V = _qkv64(0xC48)
    p_bytes = _km64_p_file({h: _km64_raw(inst, tmp_path, Q, K, h) for h in range(km64.N_HEAD)})
    got = _bcast48(inst, tmp_path, p_bytes, V, "chained")
    # S[p,h][key, query] = sum_c Q[p,h,c,query] K[p,h,c,key];
    # O[b, t, query] = sum_key S[b][key, query] V[b, t, key].
    S = np.einsum("phcq,phck->phkq", Q.astype(np.float64), K.astype(np.float64))
    expected = np.einsum("bkq,btk->btq", S.reshape(bc48.N_BLOCK, km64.N, km64.N),
                         V.astype(np.float64))
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-2, err_msg=(
        "chained attn_scores_km_64x48 -> attn_v_bcast_48 mismatch vs independent "
        "two-stage numpy reference"))


def test_seam_km_bcast_48_row_pitch_mismatch_is_caught(inst, tmp_path):
    """Mutation check: re-pack the captured producer output at the wrong
    256 B (N*ELEM_BYTES) key pitch instead of whole 512 B rows, stage it
    exactly as attn_v_bcast_48 stages p_path, and confirm the result moves by
    more than a rounding-level amount -- this seam catches a producer/consumer
    key-pitch mismatch between km_64x48 and bcast_48."""
    Q, K, V = _qkv64(0xC49)
    head_outputs = {h: _km64_raw(inst, tmp_path, Q, K, h) for h in range(km64.N_HEAD)}
    correct = _km64_p_file(head_outputs)
    cropped = b"".join(
        np.frombuffer(raw, dtype=np.float32).reshape(km64.P * km64.N, LANES)[:, :km64.N].tobytes()
        for raw in head_outputs.values())
    mispitched = cropped + bytes(len(correct) - len(cropped))
    diff = np.abs(_bcast48(inst, tmp_path, correct, V, "correct")
                  - _bcast48(inst, tmp_path, mispitched, V, "mispitched"))
    assert float(diff.max()) > 1e-2, (
        "mutating the producer's row pitch to the old buggy 256 B stride did not "
        f"change attn_v_bcast_48's output (max diff {float(diff.max()):.3e})")


# -- softmax feed: L5 scores into the real softmax kernels --------------------
#
# The question these tests answer: can a producer's raw,
# UNCROPPED output (whole 512 B rows, N_TOK live lanes) be fed straight into
# softmax? The softmax harnesses take a dense row-major (rows x cols) file and
# refuse any other size, so verbatim it is rejected rather than silently read
# as garbage; cropped to the live lanes (the consumer-crops convention) it
# gives the right softmax. Softmax kernels live in another family, so these
# need its .asm files as data of this target.

_SOFTMAX_ASM = [files(f"ipu_apps.kernels.softmax.{k}").joinpath(f"{k}.asm")
                for k in ("softmax_rows_partial", "softmax_columns_packed")]
needs_softmax = pytest.mark.skipif(
    not all(p.is_file() for p in _SOFTMAX_ASM),
    reason="the softmax kernels' .asm files are not in this target's runfiles")


def _softmax(x, axis):
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


@needs_softmax
def test_query_major_scores_feed_softmax_rows(inst, tmp_path):
    from ipu_apps.kernels.softmax.cases import run_array
    from ipu_apps.kernels.softmax.softmax_rows_partial.app import SoftmaxRowsPartialApp

    N, D = qk16.N, qk16.D
    rng = np.random.RandomState(0)
    Q = (rng.randn(D, N) * 2).astype(np.float32)
    K = (rng.randn(D, N) * 2).astype(np.float32)
    raw = _qk16_raw(inst, tmp_path, Q, K, "softmax")
    assert len(raw) == N * ROW_BYTES

    verbatim = tmp_path / "uncropped.bin"
    verbatim.write_bytes(raw)
    with pytest.raises(ValueError, match="input must hold"):
        SoftmaxRowsPartialApp(inst_path=assemble_kernel("softmax_rows_partial", tmp_path),
                              input_path=verbatim, output_path=tmp_path / "sm.bin",
                              n=N, rows=N).run(max_cycles=2_000_000)

    scores = np.frombuffer(raw, dtype=np.float32).reshape(N, LANES)[:, :N]
    _, got = run_array("softmax_rows_partial", np.ascontiguousarray(scores), axis=1)
    err = float(np.max(np.abs(got - _softmax(Q.T @ K, axis=1))))
    assert err < 1e-4, f"softmax of cropped qk_scores_16x60 output: max abs error {err:.3e}"


@needs_softmax
def test_key_major_scores_feed_softmax_columns(inst, tmp_path):
    from ipu_apps.kernels.softmax.cases import run_array
    from ipu_apps.kernels.softmax.softmax_columns_packed.app import SoftmaxColumnsPackedApp

    N, D, head = km16.N_TOK, km16.D, 1
    rng = np.random.RandomState(1)
    Q = rng.uniform(-1.0, 1.0, size=(km16.N_HEADS * D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(km16.N_HEADS * D, N)).astype(np.float32)
    raw = _km16_raw(inst, tmp_path, Q, K, head, "softmax")
    assert len(raw) == N * km16.N_TG * ROW_BYTES

    verbatim = tmp_path / "uncropped.bin"
    verbatim.write_bytes(raw)
    with pytest.raises(ValueError, match="input must hold"):
        SoftmaxColumnsPackedApp(inst_path=assemble_kernel("softmax_columns_packed", tmp_path),
                                input_path=verbatim, output_path=tmp_path / "smc.bin",
                                rows=N, width=N).run(max_cycles=2_000_000)

    # Key-major storage: row s = key s. Softmax reduces over keys, i.e. down
    # each COLUMN of the [key, query] matrix.
    scores_km = np.frombuffer(raw, dtype=np.float32).reshape(N, LANES)[:, :N]
    _, got = run_array("softmax_columns_packed", np.ascontiguousarray(scores_km), axis=0)
    lo = head * D
    expected = _softmax((Q[lo:lo + D].T @ K[lo:lo + D]).T, axis=0)      # [key, query]
    err = float(np.max(np.abs(got - expected)))
    assert err < 1e-4, f"softmax of cropped attn_scores_km_16x60 output: max abs error {err:.3e}"
