"""Seam test: attn_scores_km_16x60 -> attn_v_bcast_60 (Layer 5, KEY-MAJOR chain).

Runs the REAL producer kernel (``attn_scores_km_16x60``) to completion, takes
its raw XMEM store output --- the actual bytes ``teardown()`` wrote via
``dump_xmem_to_binary`` --- and feeds that file byte-for-byte as the consumer
kernel's (``attn_v_bcast_60``) P input, verbatim (no reshape beyond what
``attn_v_bcast_60.setup()`` itself does).

This is the SAME kernel-shape class (N<=LANES, single token group, key-major
restage) that produced the documented attn_scores_km_64x48/attn_v_bcast_48
row-pitch bug (kernel_docs/kernel_layer_map.md, "Crop convention" section):
attn_scores_km_64x48 used to crop its key rows to N*ELEM_BYTES before storing
them, while attn_v_bcast_48 addressed one key per WHOLE 512B row -- every key
row after the first landed at the wrong offset. Both attn_scores_km_16x60 and
attn_v_bcast_60 are already listed as using the FULL-ROW convention in that
doc, so this test's main job is confirming that holds for the real store/load
path, not just the documented intent.

Chain shape:
    attn_scores_km_16x60 -> S[i, s] key-major, one head per run, one row per
                             key s (lane i = query i).
    attn_v_bcast_60      -> expects P key-major: P[h, s, :N_TOK] = "row s =
                             all queries for key s", per head, heads
                             concatenated at PBASE + h*P_HEAD_STRIDE.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_scores_km_16x60 import (
    AttnScoresKM16x60App, N_TOK as KM_N_TOK, D as KM_D, N_TG as KM_N_TG,
    N_HEADS, LANES,
)
from ipu_apps.attn_v_bcast_60 import (
    AttnVBcast60App, N_TOK, D, N_HEAD, N_CHAN,
    PBASE, VBASE, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS, P_HEAD_STRIDE,
    P_ROWS, ROW_BYTES,
)

_KM_INST_BIN = Path(os.environ["ATTN_SCORES_KM_16X60_INST_BIN"])
_BCAST_INST_BIN = Path(os.environ["ATTN_V_BCAST_60_INST_BIN"])

assert KM_N_TOK == N_TOK and KM_D == D and KM_N_TG == 1 and N_HEADS == N_HEAD, (
    "attn_scores_km_16x60 / attn_v_bcast_60 shape constants diverged"
)

_POISON = np.float32(1e3)


def _run_km_scores(tmp_path: Path, Q: np.ndarray, K: np.ndarray, head: int, tag: str) -> bytes:
    """Run the REAL attn_scores_km_16x60 kernel for one head; return raw output bytes.

    Q, K are the FULL canonical [N_HEADS*D, N_TOK] channel-major inputs --
    attn_scores_km_16x60 itself slices out `head`'s channel block (this
    mirrors its real interface: one 4-head input file, a `head` selector).

    Poisons the producer's ENTIRE output (S) region with 1e3 before
    setup/run touches it, so any row the kernel fails to write shows up as
    1e3 in the returned bytes.
    """
    from ipu_apps.attn_scores_km_16x60 import SBASE, OUTPUT_ROW_BYTES, N_TOK as _KM_N_TOK, N_TG as _KM_N_TG

    q_path = tmp_path / f"q_{tag}.bin"
    k_path = tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / f"s_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    poison_row = np.full(LANES, _POISON, dtype=np.float32).tobytes()
    for r in range(_KM_N_TOK * _KM_N_TG):
        state.xmem.write_address(SBASE + r * OUTPUT_ROW_BYTES, bytearray(poison_row))

    app = AttnScoresKM16x60App(
        inst_path=_KM_INST_BIN,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=head,
    )
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = output_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    assert not np.any(arr == _POISON), (
        f"{tag}: producer under-wrote its output region -- poison value "
        f"{_POISON} survived into the stored bytes"
    )
    return raw


def _run_attn_v_bcast(tmp_path: Path, p_bytes: bytes, v_buf: np.ndarray, tag: str) -> np.ndarray:
    """Run the REAL attn_v_bcast_60 kernel with a pre-built raw P byte blob.

    Poisons the CONSUMER's entire O (output) region with 1e3 before
    setup/run, so any channel row attn_v_bcast_60 fails to write shows up as
    1e3 in the returned array.
    """
    from ipu_apps.attn_v_bcast_60 import OBASE, O_CHAN_BYTES

    p_path = tmp_path / f"p_{tag}.bin"
    v_path = tmp_path / f"v_{tag}.bin"
    p_path.write_bytes(p_bytes)
    v_path.write_bytes(v_buf.tobytes())
    output_path = tmp_path / f"o_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    poison_row = np.full(LANES, _POISON, dtype=np.float32).tobytes()
    for c in range(N_CHAN):
        state.xmem.write_address(OBASE + c * O_CHAN_BYTES, bytearray(poison_row))

    app = AttnVBcast60App(
        inst_path=_BCAST_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw_out = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert not np.any(raw_out.reshape(N_CHAN, LANES)[:, :N_TOK] == _POISON), (
        f"{tag}: attn_v_bcast_60 under-wrote its output region -- poison value "
        f"{_POISON} survived into the valid-lane bytes"
    )
    return raw_out.reshape(N_CHAN, LANES)


def _build_chained_p(tmp_path: Path, Q: np.ndarray, K: np.ndarray) -> bytes:
    """Run attn_scores_km_16x60 once per head (head selects the channel
    slice out of the SAME 4-head Q/K files) and concatenate raw outputs in
    head order -- the byte layout attn_v_bcast_60.setup() writes verbatim at
    PBASE. Q, K are [N_HEAD*D, N_TOK] canonical channel-major.
    """
    parts = []
    for h in range(N_HEAD):
        raw = _run_km_scores(tmp_path, Q, K, head=h, tag=f"h{h}")
        assert len(raw) == P_HEAD_STRIDE_ROWS * ROW_BYTES, (
            f"head {h}: attn_scores_km_16x60 produced {len(raw)} B, "
            f"expected {P_HEAD_STRIDE_ROWS * ROW_BYTES} B (P_HEAD_STRIDE)"
        )
        parts.append(raw)
    return b"".join(parts)


def _acc_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Mirrors attn_v_bcast_60's MULT + ACC.ADD float32 left-fold (per
    test_attn_v_bcast_60_wide.py's _acc_reference) -- NOT the AGG float64
    fold used by the query-major chain's reference.
    """
    out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            acc = np.zeros(N_TOK, dtype=np.float32)
            for s in range(N_TOK):
                prod = (P[h, :, s].astype(np.float32) * np.float32(V[h, t, s]))
                acc = (acc + prod).astype(np.float32) if s else prod.astype(np.float32)
            out[h, :, t] = acc
    return out


def test_seam_km_to_bcast_60_agrees(tmp_path: Path) -> None:
    """Full chain: poison attn_v_bcast_60's P region, run
    attn_scores_km_16x60 for real (x4 heads), stage its raw output verbatim
    as P, run attn_v_bcast_60 for real, and compare against a numpy reference
    of the WHOLE two-stage computation using the BROADCAST (ACC.ADD) fold --
    never the AGG fold from the sibling chain.
    """
    rng = np.random.RandomState(0x5B10)

    n_chan = N_HEAD * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_bytes = _build_chained_p(tmp_path, Q, K)
    assert len(p_bytes) == P_ROWS * ROW_BYTES, (
        f"chained P is {len(p_bytes)} B, attn_v_bcast_60 expects "
        f"{P_ROWS * ROW_BYTES} B (P_ROWS={P_ROWS} rows) -- "
        f"producer/consumer region-size mismatch"
    )

    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(P_ROWS, LANES)
    assert not np.any(p_arr[:, :N_TOK] == _POISON), (
        "producer failed to overwrite poison in its declared-valid lanes"
    )

    # V is channel-major, one WHOLE row per value channel -- host-staged
    # exactly as attn_v_bcast_60's own isolated test builds it (V has no
    # producer kernel in this repo, so its padding/reshape is not part of
    # the seam under test).
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    got = _run_attn_v_bcast(tmp_path, p_bytes, v_buf, tag="chained")

    # Full two-stage numpy reference: key-major QKt per head, then the
    # broadcast ACC.ADD fold (not qk's AGG fold).
    S = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float32)  # [h, query i, key s]
    for h in range(N_HEAD):
        lo = h * D
        q_head = Q[lo:lo + D]
        k_head = K[lo:lo + D]
        S[h] = q_head.T @ k_head

    expected = _acc_reference(S, V)

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-4, atol=1e-3,
                err_msg=(
                    f"chained attn_scores_km_16x60 -> attn_v_bcast_60 "
                    f"mismatch (head {h}, channel {t}) against the full "
                    f"two-stage numpy reference"
                ),
            )


def test_seam_km_to_bcast_60_row_pitch_and_head_concat(tmp_path: Path) -> None:
    """Assert on RAW STORED BYTES (not cropped/reshaped arrays) that:
      1. each key's row is a WHOLE ROW_BYTES=512B (row pitch), matching the
         attn_v_bcast_60 addressing convention (one key per whole row) --
         this is exactly the axis the attn_scores_km_64x48/attn_v_bcast_48
         bug broke (256B crop vs 512B stride expectation).
      2. concatenating N_HEAD single-head producer runs in head order lands
         each head's block at PBASE + h*P_HEAD_STRIDE, matching
         attn_v_bcast_60's own indexing (attn_v_bcast_60/__init__.py
         P_HEAD_STRIDE_ROWS/cr7).

    Uses head-identifiable data (head h's Q/K filled with h+1) so raw score
    magnitude alone identifies which head's block landed where, independent
    of any numeric tolerance.
    """
    Q = np.zeros((N_HEAD * D, N_TOK), dtype=np.float32)
    K = np.zeros((N_HEAD * D, N_TOK), dtype=np.float32)
    for h in range(N_HEAD):
        Q[h * D:(h + 1) * D, :] = float(h + 1)
        K[h * D:(h + 1) * D, :] = 1.0  # S = D * (h+1), uniform across (i, s)

    # Single-head raw output, checked BEFORE concatenation: row pitch must be
    # exactly ROW_BYTES per key, N_TOK keys, i.e. len == N_TOK * ROW_BYTES.
    raw_h0 = _run_km_scores(tmp_path, Q, K, head=0, tag="pitchcheck")
    assert len(raw_h0) == N_TOK * ROW_BYTES, (
        f"attn_scores_km_16x60 raw output is {len(raw_h0)} B for {N_TOK} keys "
        f"-- expected exactly {N_TOK * ROW_BYTES} B ({ROW_BYTES} B/row, full "
        f"row pitch). A narrower (e.g. N_TOK*4 B/row) pitch here would be "
        f"the same class of bug that broke attn_scores_km_64x48/"
        f"attn_v_bcast_48."
    )
    arr_h0 = np.frombuffer(raw_h0, dtype=np.float32).reshape(N_TOK, LANES)
    # Row s, lane i should equal D*(1) = D for every (i, s) since Q=K=1 for head 0.
    np.testing.assert_allclose(arr_h0[:, :N_TOK], float(D), rtol=1e-5)

    # Now the full chained blob, concatenated in head order.
    p_bytes = _build_chained_p(tmp_path, Q, K)
    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(N_HEAD, P_HEAD_STRIDE_ROWS, LANES)

    for h in range(N_HEAD):
        block = p_arr[h, :, :N_TOK]
        expected_val = np.float32(D * (h + 1))
        np.testing.assert_allclose(
            block, expected_val, rtol=1e-5,
            err_msg=(
                f"block at head-slot {h} (byte offset {h * P_HEAD_STRIDE}) "
                f"does not contain head {h}'s scores -- head concatenation "
                f"order does not match attn_v_bcast_60's "
                f"PBASE + h*P_HEAD_STRIDE indexing (cr7=P_HEAD_STRIDE_ROWS)"
            ),
        )


def test_seam_km_to_bcast_60_mutation_kills_test(tmp_path: Path) -> None:
    """Harness-teeth check: shift the chained P blob by one row (mimicking
    the exact documented attn_scores_km_64x48/attn_v_bcast_48 pitch-mismatch
    bug class) and confirm the real-reference comparison FAILS. Self-check on
    the harness, not a recorded product defect -- reverted, not shipped as-is.
    """
    rng = np.random.RandomState(0x5B11)
    n_chan = N_HEAD * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_bytes = _build_chained_p(tmp_path, Q, K)

    # Mutate: shift every row's base by one ROW_BYTES (drop first row, pad a
    # garbage row at the end) -- the same class as the documented
    # attn_scores_km_64x48/attn_v_bcast_48 bug.
    mutated = bytearray(p_bytes[ROW_BYTES:])
    mutated.extend(np.full(LANES, _POISON, dtype=np.float32).tobytes())
    assert len(mutated) == len(p_bytes)

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    got = _run_attn_v_bcast(tmp_path, bytes(mutated), v_buf, tag="mutated")

    S = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float32)
    for h in range(N_HEAD):
        lo = h * D
        S[h] = Q[lo:lo + D].T @ K[lo:lo + D]
    expected = _acc_reference(S, V)

    with pytest.raises(AssertionError):
        for h in range(N_HEAD):
            for t in range(D):
                np.testing.assert_allclose(
                    got[h * D + t, :N_TOK], expected[h, :, t],
                    rtol=1e-4, atol=1e-3,
                )
