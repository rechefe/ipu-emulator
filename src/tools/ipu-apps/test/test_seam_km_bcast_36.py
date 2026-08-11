"""Seam test: attn_scores_km_256x36 -> attn_v_bcast_36 (key-major, no-AGG
broadcast chain, L3).

Nothing today feeds `attn_scores_km_256x36`'s REAL XMEM store output,
byte-for-byte, as `attn_v_bcast_36`'s `p_path`. This test closes that gap:
run `attn_scores_km_256x36` once per head (it takes `head=` and slices a
canonical [N_HEADS*D, N_TOK] Q/K file itself), concatenate the four per-head
raw output files in head order (h=0..3), and feed the concatenation verbatim
as `attn_v_bcast_36`'s `p_path`, with no reshape/crop in between.

Per kernel_docs/kernel_layer_map.md and both kernels' `__init__.py`:
  attn_scores_km_256x36 per-head output: N_TOK * N_TG = 512 rows, row (s, g)
    at SBASE_ROW + s*N_TG + g  (key-major: lane i = query, row = key s /
    group g).
  attn_v_bcast_36 P input: P[i, s] at PBASE + h*P_HEAD_STRIDE_ROWS(=512) +
    s*PV_STRIDE_ROWS(=2) + i//128 rows (4 heads, head-major); PV_STRIDE_ROWS
    == N_TG == 2, so the intra-block row formula (s*2 + g) is identical to
    attn_scores_km's stored layout. This test verifies that empirically.

This chain uses ACC.ADD (no AGG, no cross-chunk carry) -- must NOT be mixed
with the query-major + AGG chain (separate goldens, bit-different results by
design; see kernel_layer_map.md "Two attention mappings -- never mix them").
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_scores_km_256x36 import (
    AttnScoresKM256x36App, N_TOK, D, N_TG, N_TPG, N_HEADS, LANES,
    SBASE, OUTPUT_ROW_BYTES,
)
from ipu_apps.attn_v_bcast_36 import (
    AttnVBcast36App, N_HEAD, N_CHAN, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS,
    OBASE, O_CHAN_BYTES,
)

_KM_INST_BIN = Path(os.environ["ATTN_SCORES_KM_256X36_INST_BIN"])
_BCAST_INST_BIN = Path(os.environ["ATTN_V_BCAST_36_INST_BIN"])

POISON = np.float32(1e3)

assert N_HEADS == N_HEAD
assert PV_STRIDE_ROWS == N_TG


def _run_attn_scores_km(tmp_path: Path, q_path: Path, k_path: Path, head: int, tag: str) -> bytes:
    """Run attn_scores_km_256x36 for one head and return its raw output bytes."""
    output_path = tmp_path / f"km_out_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    # Poison the destination region before the real run.
    poison_row = np.full(LANES, POISON, dtype=np.float32).tobytes()
    for r in range(N_TOK * N_TG):
        state.xmem.write_address(SBASE + r * OUTPUT_ROW_BYTES, bytearray(poison_row))

    app = AttnScoresKM256x36App(
        inst_path=_KM_INST_BIN,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=head,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    assert len(raw) == N_TOK * N_TG * OUTPUT_ROW_BYTES
    arr = np.frombuffer(raw, dtype=np.float32)
    assert not np.any(arr == POISON), (
        f"{tag}: producer under-wrote its output region -- poison value "
        f"{POISON} survived into the stored bytes"
    )
    return raw


def test_seam_attn_scores_km_to_attn_v_bcast_36(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x6EA3)

    n_chan = N_HEADS * D
    # Canonical channel-major files consumed directly by attn_scores_km_256x36
    # (element [token t, channel h*D+c] at (h*D+c)*N_TOK + t).
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q_km.bin"
    k_path = tmp_path / "k_km.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    # --- Stage 1: run attn_scores_km_256x36 once per head, concatenate raw
    # bytes in head order (h=0..3) -- what attn_v_bcast_36's P layout expects. ---
    per_head_raw = [
        _run_attn_scores_km(tmp_path, q_path, k_path, head=h, tag=f"h{h}")
        for h in range(N_HEADS)
    ]
    p_bytes = b"".join(per_head_raw)
    assert len(p_bytes) == N_HEAD * P_HEAD_STRIDE_ROWS * 512, (
        "concatenated attn_scores_km output size does not equal "
        "attn_v_bcast_36's expected P region size -- head-block row counts disagree"
    )

    p_path = tmp_path / "p_chained_km.bin"
    p_path.write_bytes(p_bytes)

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]
    v_path = tmp_path / "v_bcast.bin"
    v_path.write_bytes(v_buf.tobytes())

    # --- Stage 2: poison attn_v_bcast_36's output region, then run it on the
    # chained P bytes verbatim (no reshape/crop). ---
    output_path = tmp_path / "attn_v_bcast_out.bin"
    state2 = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    poison_row = np.full(LANES, POISON, dtype=np.float32).tobytes()
    for c in range(N_CHAN):
        for r in range(O_CHAN_BYTES // 512):
            state2.xmem.write_address(OBASE + c * O_CHAN_BYTES + r * 512, bytearray(poison_row))

    app2 = AttnVBcast36App(
        inst_path=_BCAST_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    state2, cycles2 = app2.run(max_cycles=20_000_000, state=state2)
    assert cycles2 > 0

    raw_out = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw_out.size == N_CHAN * 2 * LANES
    assert not np.any(raw_out == POISON), (
        "attn_v_bcast_36 under-wrote its output region when fed chained km P"
    )
    got = raw_out.reshape(N_CHAN, 2 * LANES)

    # --- Numeric agreement: whole two-stage numpy reference. Scores are raw
    # S[s,i] = sum_c Q[i,c]*K[s,c] per head from attn_scores_km, fed straight
    # through as P for attn_v_bcast: O[i,t] = sum_s P[i,s]*V[s,t]. ---
    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float64)
    for h in range(N_HEAD):
        lo = h * D
        q_head = Q[lo:lo + D].astype(np.float64)   # [D, N_TOK]
        k_head = K[lo:lo + D].astype(np.float64)   # [D, N_TOK]
        S = q_head.T @ k_head                        # S[i, s] = sum_c Q[i,c]*K[s,c]
        expected[h] = S @ V[h].T.astype(np.float64)   # [N_TOK, D]

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-3, atol=1e-2,
                err_msg=f"chained attn_scores_km->attn_v_bcast mismatch for head {h}, channel {t}",
            )


def test_seam_km_bcast_head_block_pitch_mutation_detected(tmp_path: Path) -> None:
    """Harness-teeth check: corrupt one head's block placement, run it
    through the REAL attn_v_bcast_36 kernel, and confirm the numeric
    agreement assertion actually fails. Reverted after; not shipped passing.
    """
    rng = np.random.RandomState(0x6EA4)
    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q_km_mut.bin"
    k_path = tmp_path / "k_km_mut.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    per_head_raw = [
        _run_attn_scores_km(tmp_path, q_path, k_path, head=h, tag=f"mut{h}")
        for h in range(N_HEADS)
    ]
    good = b"".join(per_head_raw)

    # Mutate: roll head 1's block by one row (512 B) -- the exact class of
    # bug this audit was chartered to find (attn_scores_km_64x48 sub-row
    # pitch vs attn_v_bcast_48's whole-row read).
    row = 512
    mutated = bytearray(good)
    h_mut = 1
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
    v_path = tmp_path / "v_bcast_mut.bin"
    v_path.write_bytes(v_buf.tobytes())

    p_path = tmp_path / "p_mutated_km.bin"
    p_path.write_bytes(bytes(mutated))
    output_path = tmp_path / "attn_v_bcast_out_mut.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnVBcast36App(
        inst_path=_BCAST_INST_BIN,
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
        lo = h * D
        q_head = Q[lo:lo + D].astype(np.float64)
        k_head = K[lo:lo + D].astype(np.float64)
        S = q_head.T @ k_head
        expected[h] = S @ V[h].T.astype(np.float64)

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
