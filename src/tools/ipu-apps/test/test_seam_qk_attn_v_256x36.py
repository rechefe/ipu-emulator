"""Seam test: qk_scores_256x36 -> attn_v_256x36 (query-major + AGG chain, L3).

Nothing today feeds `qk_scores_256x36`'s REAL XMEM store output, byte-for-byte,
as `attn_v_256x36`'s `p_path`. Each kernel's own test hand-stages numpy arrays
into its own harness-specific layout. This test closes that gap for the
query-major chain: run `qk_scores_256x36` once per head (it has no internal
head slicing -- see its `__init__.py`, it only ever consumes a single D=36
channel-major Q/K pair), concatenate the four per-head raw output files in
head order (h=0..3), and feed the concatenation verbatim as `attn_v_256x36`'s
`p_path`, with no reshape/crop in between.

Per kernel_docs/kernel_layer_map.md:
  qk_scores_256x36 per-head output: N * N_TG = 512 rows, row (i, g) at
    S_BASE_ROW + i*N_TG + g  (query-major, group-interleaved).
  attn_v_256x36 P input: P[i, s] at PBASE + h*P_HEAD_STRIDE_ROWS(=512)
    + i*PV_STRIDE_ROWS(=2) + s//128 rows (4 heads, head-major).

P_HEAD_STRIDE_ROWS (512) equals qk_scores's single-head output row count
(512) and the intra-block addressing formula (i*2 + g) is identical on both
sides, so four concatenated single-head qk_scores runs *should* line up
exactly with attn_v_256x36's expected P layout -- this test verifies that
empirically rather than by re-deriving the arithmetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_256x36 import QkScores256x36App, N, D, N_TG, N_TPG, LANES
from ipu_apps.attn_v_256x36 import (
    AttnV256x36App, N_TOK, N_HEAD, N_CHAN, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
)

_QK_INST_BIN = Path(os.environ["QK_SCORES_256X36_INST_BIN"])
_ATTN_V_INST_BIN = Path(os.environ["ATTN_V_256X36_INST_BIN"])

POISON = np.float32(1e3)

assert N == N_TOK
assert D * N_HEAD == N_CHAN


def _run_qk_scores(tmp_path: Path, q_head: np.ndarray, k_head: np.ndarray, tag: str) -> bytes:
    """Run qk_scores_256x36 for one head's [D, N] Q/K and return its raw output bytes."""
    q_path = tmp_path / f"q_{tag}.bin"
    k_path = tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(q_head.astype(np.float32).tobytes())
    k_path.write_bytes(k_head.astype(np.float32).tobytes())
    output_path = tmp_path / f"qk_out_{tag}.bin"

    # Poison the destination region before the real run so any under-write
    # by the producer (or any byte the consumer reads that the producer
    # never actually wrote) is detectable: pre-fill the file slot with a
    # recognizable value, then overwrite via a fresh XMem write pass.
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Poison the whole S region (N * N_TG rows) before setup/run touches it.
    from ipu_apps.qk_scores_256x36 import S_BASE, OUTPUT_ROW_BYTES
    poison_row = (np.full(LANES, POISON, dtype=np.float32)).tobytes()
    for r in range(N * N_TG):
        state.xmem.write_address(S_BASE + r * OUTPUT_ROW_BYTES, bytearray(poison_row))

    app = QkScores256x36App(
        inst_path=_QK_INST_BIN,
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    assert len(raw) == N * N_TG * OUTPUT_ROW_BYTES
    arr = np.frombuffer(raw, dtype=np.float32)
    # Poison detector: no output element should still read as the poison
    # value (extremely unlikely from real uniform(-1,1) products/sums).
    assert not np.any(arr == POISON), (
        f"{tag}: producer under-wrote its output region -- poison value "
        f"{POISON} survived into the stored bytes"
    )
    return raw


def test_seam_qk_scores_to_attn_v_256x36(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x5EA3)

    # Canonical channel-major multi-head Q/K, [N_HEAD*D, N_TOK].
    Q = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # --- Stage 1: run qk_scores_256x36 once per head, concatenate raw bytes
    # in head order (h=0..3) -- exactly what attn_v_256x36's P layout expects. ---
    per_head_raw = []
    for h in range(N_HEAD):
        raw = _run_qk_scores(tmp_path, Q[h], K[h], tag=f"h{h}")
        per_head_raw.append(raw)

    p_bytes = b"".join(per_head_raw)
    assert len(p_bytes) == N_HEAD * P_HEAD_STRIDE_ROWS * 512, (
        "concatenated qk_scores output size does not equal attn_v_256x36's "
        "expected P region size -- head-block row counts disagree"
    )

    p_path = tmp_path / "p_chained.bin"
    p_path.write_bytes(p_bytes)

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]
    v_path = tmp_path / "v.bin"
    v_path.write_bytes(v_buf.tobytes())

    # --- Stage 2: poison attn_v_256x36's output region, then run it on the
    # chained P bytes verbatim (no reshape/crop). ---
    output_path = tmp_path / "attn_v_out.bin"
    state2 = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    from ipu_apps.attn_v_256x36 import OBASE, O_CHAN_BYTES
    poison_row = (np.full(LANES, POISON, dtype=np.float32)).tobytes()
    for c in range(N_CHAN):
        for r in range(O_CHAN_BYTES // 512):
            state2.xmem.write_address(OBASE + c * O_CHAN_BYTES + r * 512, bytearray(poison_row))

    app2 = AttnV256x36App(
        inst_path=_ATTN_V_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state2, cycles2 = app2.run(max_cycles=20_000_000, state=state2)
    assert cycles2 > 0

    raw_out = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw_out.size == N_CHAN * 2 * LANES
    assert not np.any(raw_out == POISON), (
        "attn_v_256x36 under-wrote its output region when fed chained qk_scores P"
    )
    got = raw_out.reshape(N_CHAN, 2 * LANES)

    # --- Numeric agreement: whole two-stage numpy reference, not each
    # kernel's own isolated golden. Scores S = Q^T @ K per head (raw scores,
    # no softmax -- qk_scores stores raw scores; attn_v_256x36 treats its P
    # input as already-softmaxed attention weights, so here we feed raw
    # scores through both stages symmetrically: the chain under test is
    # "whatever qk_scores emits is what attn_v consumes as P", regardless of
    # whether a softmax would normally sit in between). ---
    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float64)
    for h in range(N_HEAD):
        S = Q[h].T.astype(np.float64) @ K[h].astype(np.float64)     # [N_TOK, N_TOK]
        expected[h] = S @ V[h].T.astype(np.float64)                  # [N_TOK, D]

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-3, atol=1e-2,
                err_msg=f"chained qk_scores->attn_v mismatch for head {h}, channel {t}",
            )


def test_seam_qk_attn_v_head_block_pitch_mutation_detected(tmp_path: Path) -> None:
    """Harness-teeth check: corrupt one head's block placement, run it
    through the REAL attn_v_256x36 kernel, and confirm the numeric-agreement
    assertion (the one the main seam test relies on) actually fails.

    This proves the seam test isn't vacuously passing -- it isolates exactly
    the claim in the kernel_layer_map.md bug writeup: shifting a producer
    block's row pitch/base must be detectable by comparing the chained
    kernel output to the two-stage numpy reference. Reverted after; the
    mutated version is never shipped as a passing assertion.
    """
    rng = np.random.RandomState(0x5EA4)
    Q = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    per_head_raw = [
        _run_qk_scores(tmp_path, Q[h], K[h], tag=f"mut{h}") for h in range(N_HEAD)
    ]
    good = b"".join(per_head_raw)

    # Mutate: roll head 2's block by one row (512 B) -- simulate a producer
    # that shifted its row pitch/base by one row, analogous to the
    # attn_scores_km_64x48 / attn_v_bcast_48 sub-row-pitch bug.
    row = 512
    mutated = bytearray(good)
    h_mut = 2
    block_start = h_mut * P_HEAD_STRIDE_ROWS * row
    block_end = block_start + P_HEAD_STRIDE_ROWS * row
    block = bytearray(mutated[block_start:block_end])
    rolled = block[row:] + block[:row]
    mutated[block_start:block_end] = rolled
    assert bytes(mutated) != good, "mutation harness produced no actual change"

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]
    v_path = tmp_path / "v_mut.bin"
    v_path.write_bytes(v_buf.tobytes())

    p_path = tmp_path / "p_mutated.bin"
    p_path.write_bytes(bytes(mutated))
    output_path = tmp_path / "attn_v_out_mut.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnV256x36App(
        inst_path=_ATTN_V_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw_out = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw_out.reshape(N_CHAN, 2 * LANES)

    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float64)
    for h in range(N_HEAD):
        S = Q[h].T.astype(np.float64) @ K[h].astype(np.float64)
        expected[h] = S @ V[h].T.astype(np.float64)

    # The correct chain (test_seam_qk_scores_to_attn_v_256x36) passes this
    # exact assertion shape for every head/channel. With head 2's block
    # pitch-rolled, at least one (t) comparison for h=2 must now fail --
    # proving the assertion has teeth and isn't loose enough to pass anyway.
    mismatches = 0
    for t in range(D):
        try:
            np.testing.assert_allclose(
                got[h_mut * D + t, :N_TOK], expected[h_mut, :, t],
                rtol=1e-3, atol=1e-2,
            )
        except AssertionError:
            mismatches += 1
    assert mismatches > 0, (
        "mutated (row-rolled) head block did not produce any numeric "
        "mismatch -- the seam assertion would not have caught this bug"
    )
