"""Seam test: qk_scores_16x60 -> attn_v_16x60 (Layer 5, QUERY-MAJOR chain).

Unlike the per-kernel wide tests (which hand-stage numpy arrays directly into
each kernel's expected in-memory layout), this test runs the REAL producer
kernel (``qk_scores_16x60``) to completion, takes its raw XMEM store output
--- the actual bytes ``teardown()`` wrote via ``dump_xmem_to_binary`` --- and
feeds that file byte-for-byte as the consumer kernel's (``attn_v_16x60``) P
input, with no reshaping beyond what ``attn_v_16x60.setup()`` itself does
(``write_address`` of the raw file contents, verbatim).

Chain shape (see kernel_docs/kernel_layer_map.md "Two attention mappings"):
    qk_scores_16x60  -> S[i, s] query-major, one head per run, one row per
                         query i (lane s = key s).
    attn_v_16x60     -> expects P query-major: P[h, i, :N_TOK] = "row i = all
                         keys for query i", per head, heads concatenated at
                         PBASE + h*P_HEAD_STRIDE.

qk_scores_16x60 computes ONE head per invocation, so this test runs it 4
times (once per head, on that head's Q/K channel slice) and concatenates the
4 raw output files in head order at PBASE + h*P_HEAD_STRIDE_ROWS*ROW_BYTES --
exactly the block index attn_v_16x60 expects. This is the seam: nothing
downstream of qk_scores_16x60 ever reshapes/crops the file before it lands in
attn_v_16x60's XMEM.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_16x60 import (
    QkScores16x60App, N as QK_N, D as QK_D, N_TG as QK_N_TG, LANES,
)
from ipu_apps.attn_v_16x60 import (
    AttnV16x60App, N_TOK, D, N_HEAD, N_CHAN,
    PBASE, VBASE, PV_STRIDE_ROWS, P_HEAD_STRIDE_ROWS, P_HEAD_STRIDE,
    P_ROWS, ROW_BYTES,
)

_QK_INST_BIN = Path(os.environ["QK_SCORES_16X60_INST_BIN"])
_ATTN_V_INST_BIN = Path(os.environ["ATTN_V_16X60_INST_BIN"])

assert QK_N == N_TOK and QK_D == D and QK_N_TG == 1, (
    "qk_scores_16x60 / attn_v_16x60 shape constants diverged"
)

_POISON = np.float32(1e3)


def _run_qk_scores(tmp_path: Path, q_head: np.ndarray, k_head: np.ndarray, tag: str) -> bytes:
    """Run the REAL qk_scores_16x60 kernel for one head; return its raw output bytes.

    Poisons the producer's ENTIRE output (S) region with 1e3 before setup/run
    touches it, so any row the kernel fails to write shows up as 1e3 in the
    returned bytes -- checked by the caller.
    """
    from ipu_apps.qk_scores_16x60 import S_BASE, OUTPUT_ROW_BYTES, N as _QK_N, N_TG as _QK_N_TG

    q_path = tmp_path / f"q_{tag}.bin"
    k_path = tmp_path / f"k_{tag}.bin"
    q_path.write_bytes(q_head.tobytes())
    k_path.write_bytes(k_head.tobytes())
    output_path = tmp_path / f"s_{tag}.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    poison_row = np.full(LANES, _POISON, dtype=np.float32).tobytes()
    for r in range(_QK_N * _QK_N_TG):
        state.xmem.write_address(S_BASE + r * OUTPUT_ROW_BYTES, bytearray(poison_row))

    app = QkScores16x60App(
        inst_path=_QK_INST_BIN,
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
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


def _run_attn_v(tmp_path: Path, p_bytes: bytes, v_buf: np.ndarray, tag: str) -> np.ndarray:
    """Run the REAL attn_v_16x60 kernel with a pre-built raw P byte blob.

    Poisons the CONSUMER's entire O (output) region with 1e3 before
    setup/run, so any channel row attn_v_16x60 fails to write shows up as
    1e3 in the returned array (the caller checks the valid-lane prefix, but
    an under-write of a whole row would leave that row's valid lanes at
    1e3 too, so it is still detectable through the numeric-agreement check;
    this assert additionally proves it directly).
    """
    from ipu_apps.attn_v_16x60 import OBASE, O_CHAN_BYTES

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

    app = AttnV16x60App(
        inst_path=_ATTN_V_INST_BIN,
        p_path=p_path,
        v_path=v_path,
        output_path=output_path,
    )
    _, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw_out = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert not np.any(raw_out.reshape(N_CHAN, LANES)[:, :N_TOK] == _POISON), (
        f"{tag}: attn_v_16x60 under-wrote its output region -- poison value "
        f"{_POISON} survived into the valid-lane bytes"
    )
    return raw_out.reshape(N_CHAN, LANES)


def _build_chained_p(tmp_path: Path, Q: np.ndarray, K: np.ndarray) -> bytes:
    """Run qk_scores_16x60 once per head and concatenate raw outputs in head
    order -- exactly the byte layout attn_v_16x60.setup() writes verbatim at
    PBASE.  Q, K are [N_HEAD*D, N_TOK] canonical channel-major.
    """
    parts = []
    for h in range(N_HEAD):
        lo = h * D
        q_head = Q[lo:lo + D]   # [D, N_TOK]
        k_head = K[lo:lo + D]
        raw = _run_qk_scores(tmp_path, q_head, k_head, tag=f"h{h}")
        assert len(raw) == P_HEAD_STRIDE_ROWS * ROW_BYTES, (
            f"head {h}: qk_scores_16x60 produced {len(raw)} B, "
            f"expected {P_HEAD_STRIDE_ROWS * ROW_BYTES} B (P_HEAD_STRIDE)"
        )
        parts.append(raw)
    return b"".join(parts)


def test_seam_qk_to_attn_v_16x60_agrees(tmp_path: Path) -> None:
    """Full chain: poison attn_v_16x60's P region, run qk_scores_16x60 for
    real (x4 heads), stage its raw output verbatim as P, run attn_v_16x60 for
    real, and compare against a numpy reference of the WHOLE two-stage
    computation (QKt, then attn_v_16x60's own AGG-mirroring reference) -- not
    each kernel's isolated golden.
    """
    rng = np.random.RandomState(0x51E0)

    n_chan = N_HEAD * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    # -- Poison: each _run_qk_scores call below fills that PRODUCER's entire
    #    output (S) region with 1e3 before it runs for real, so any row it
    #    fails to write shows up as 1e3 in the returned bytes (asserted
    #    inside _run_qk_scores). The CONSUMER's P region is poisoned
    #    separately below, before attn_v_16x60 runs.
    #
    # -- Run the producer for real, 4 heads, concatenate in head order.
    p_bytes = _build_chained_p(tmp_path, Q, K)
    assert len(p_bytes) == P_ROWS * ROW_BYTES, (
        f"chained P is {len(p_bytes)} B, attn_v_16x60 expects {P_ROWS * ROW_BYTES} B "
        f"(P_ROWS={P_ROWS} rows) -- producer/consumer region-size mismatch"
    )

    # Sanity: the producer's real output must have overwritten the poison
    # value everywhere within the valid N_TOK*ELEM_BYTES prefix of every row.
    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(P_ROWS, LANES)
    assert not np.any(p_arr[:, :N_TOK] == _POISON), (
        "producer failed to overwrite poison in its declared-valid lanes"
    )

    # V is channel-major, one WHOLE row per value channel -- host-staged
    # exactly as attn_v_16x60's own isolated test builds it (V has no
    # producer kernel in this repo). This padding/reshape is NOT part of the
    # seam under test (V never passes through a producer kernel's store
    # path), so it is done here rather than via a kernel run.
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    got = _run_attn_v(tmp_path, p_bytes, v_buf, tag="chained")

    # -- Numeric agreement: full two-stage numpy reference, not the isolated
    #    per-kernel goldens.
    S = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float32)  # [h, query i, key s]
    for h in range(N_HEAD):
        lo = h * D
        q_head = Q[lo:lo + D]
        k_head = K[lo:lo + D]
        S[h] = q_head.T @ k_head

    def _agg_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
        out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
        for h in range(N_HEAD):
            for t in range(D):
                for i in range(N_TOK):
                    lanes = (P[h, i, :].astype(np.float32) * V[h, t, :].astype(np.float32))
                    total = 0.0
                    for s in range(N_TOK):
                        total += float(lanes[s])
                    out[h, i, t] = np.float32(total)
        return out

    expected = _agg_reference(S, V)

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK], expected[h, :, t],
                rtol=1e-4, atol=1e-3,
                err_msg=(
                    f"chained qk_scores_16x60 -> attn_v_16x60 mismatch "
                    f"(head {h}, channel {t}) against the full two-stage "
                    f"numpy reference"
                ),
            )


def test_seam_qk_to_attn_v_16x60_head_concat_order(tmp_path: Path) -> None:
    """Confirm concatenating N_HEAD single-head qk_scores_16x60 runs in HEAD
    ORDER produces the byte layout attn_v_16x60 actually indexes with
    (PBASE + h*P_HEAD_STRIDE). Uses head-identifiable data: head h's Q/K are
    filled with a constant equal to h+1, so head h's raw score bytes are all
    (h+1)**2 * (something) and any block-order swap is visible without
    needing full numeric agreement.
    """
    # Distinct, head-identifiable Q/K per head so raw score magnitude alone
    # identifies which head's block landed where.
    Q = np.zeros((N_HEAD * D, N_TOK), dtype=np.float32)
    K = np.zeros((N_HEAD * D, N_TOK), dtype=np.float32)
    for h in range(N_HEAD):
        Q[h * D:(h + 1) * D, :] = float(h + 1)
        K[h * D:(h + 1) * D, :] = 1.0  # keep K uniform so S = D * (h+1)

    p_bytes = _build_chained_p(tmp_path, Q, K)
    p_arr = np.frombuffer(p_bytes, dtype=np.float32).reshape(N_HEAD, P_HEAD_STRIDE_ROWS, LANES)

    for h in range(N_HEAD):
        block = p_arr[h, :, :N_TOK]
        expected_val = np.float32(D * (h + 1))
        np.testing.assert_allclose(
            block, expected_val, rtol=1e-5,
            err_msg=(
                f"block at head-slot {h} (byte offset "
                f"{h * P_HEAD_STRIDE}) does not contain head {h}'s scores -- "
                f"head concatenation order does not match attn_v_16x60's "
                f"PBASE + h*P_HEAD_STRIDE indexing"
            ),
        )


def test_seam_qk_to_attn_v_16x60_mutation_kills_test(tmp_path: Path) -> None:
    """Harness-teeth check (not shipped as a real regression assertion of a
    bug -- proves the seam test WOULD catch a pitch/base mismatch of the
    exact class kernel_docs/kernel_layer_map.md documents for
    attn_scores_km_64x48/attn_v_bcast_48).

    Deliberately shift the chained P blob by one row (mimicking a producer
    that crops/misaligns by ROW_BYTES) and confirm the comparison against the
    real, correctly-aligned reference FAILS. This is a self-check on the test
    harness, not a recorded product defect.
    """
    rng = np.random.RandomState(0x51E1)
    n_chan = N_HEAD * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    p_bytes = _build_chained_p(tmp_path, Q, K)

    # Mutate: drop the first row and pad a garbage row at the end -- shifts
    # every row's base by exactly one ROW_BYTES, the same class of bug as the
    # documented attn_scores_km_64x48/attn_v_bcast_48 pitch mismatch.
    mutated = bytearray(p_bytes[ROW_BYTES:])
    mutated.extend(np.full(LANES, _POISON, dtype=np.float32).tobytes())
    assert len(mutated) == len(p_bytes)

    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]

    got = _run_attn_v(tmp_path, bytes(mutated), v_buf, tag="mutated")

    S = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float32)
    for h in range(N_HEAD):
        lo = h * D
        S[h] = Q[lo:lo + D].T @ K[lo:lo + D]

    def _agg_reference(P: np.ndarray, V: np.ndarray) -> np.ndarray:
        out = np.zeros((N_HEAD, N_TOK, D), dtype=np.float32)
        for h in range(N_HEAD):
            for t in range(D):
                for i in range(N_TOK):
                    lanes = (P[h, i, :].astype(np.float32) * V[h, t, :].astype(np.float32))
                    total = 0.0
                    for s in range(N_TOK):
                        total += float(lanes[s])
                    out[h, i, t] = np.float32(total)
        return out

    expected = _agg_reference(S, V)

    with pytest.raises(AssertionError):
        for h in range(N_HEAD):
            for t in range(D):
                np.testing.assert_allclose(
                    got[h * D + t, :N_TOK], expected[h, :, t],
                    rtol=1e-4, atol=1e-3,
                )


def test_seam_qk_to_attn_v_16x60_shared_state_zero_copy(tmp_path: Path) -> None:
    """Same chain as ``test_seam_qk_to_attn_v_16x60_agrees``, but qk_scores_16x60
    and attn_v_16x60 run against ONE shared IpuState with base rows placed so
    each head's qk_scores_16x60 run writes its S output DIRECTLY into
    attn_v_16x60's expected P slot for that head -- no file round-trip, no
    second IpuState, no Python byte copy of the score data between the two
    kernels. Only Q/K (real inputs, never a producer kernel's output) and V
    (no producer kernel in this repo) are staged from disk, exactly as a real
    host loading initial tensors would; that is not the host-in-the-middle
    pattern this test is checking for.

    Made possible by constructor-parameterized base rows (qrow_base_row/
    s_base_row on QkScores16x60App -- k_base_row is NOT relocatable, see
    below; pbase_row/vbase_row/obase_row on AttnV16x60App) plus an optional
    p_path=None on AttnV16x60App, which skips its own P disk-staging so the
    bytes qk_scores_16x60 already wrote into the shared state's XMEM
    survive.

    NOTE: qk_scores_16x60.asm reads K_BASE from cr0, and cr0 is hardwired to
    the constant 0 by the ISA (CR_ZERO_REG_INDEX in ipu_config.py; any write
    to it is silently dropped) -- so K's base row can NEVER be relocated
    without changing which CR the .asm uses, which is out of scope. K
    therefore always occupies rows [0, K_ROWS); every other region in this
    shared state is placed at row >= K_ROWS to avoid colliding with it.
    """
    from ipu_apps.qk_scores_16x60 import K_ROWS
    from ipu_apps.attn_v_16x60 import (
        P_HEAD_STRIDE_ROWS, V_ROWS, O_ROWS, OBASE, O_CHAN_BYTES,
    )

    rng = np.random.RandomState(0x51E2)
    n_chan = N_HEAD * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    V = rng.uniform(-1.0, 1.0, size=(N_HEAD, D, N_TOK)).astype(np.float32)

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )

    # Layout (rows), all in the ONE shared state:
    #   [0 .. K_ROWS)                          K (fixed: cr0 is hardwired 0)
    #   [K_ROWS .. +P_ROWS)                    P / S  (shared: qk_scores
    #                                            writes, attn_v reads, in
    #                                            place)
    #   [.. +V_ROWS)                           V (host-staged; no producer)
    #   [.. +O_ROWS)                           O (attn_v's output)
    #   [.. +QROW_ROWS)                        qk_scores_16x60's staged-Q
    #                                            scratch (reused per head,
    #                                            never read by attn_v_16x60)
    p_base_row = K_ROWS
    v_base_row = p_base_row + P_ROWS
    o_base_row = v_base_row + V_ROWS
    qk_staging_base_row = o_base_row + O_ROWS

    p_base = p_base_row * ROW_BYTES
    v_base = v_base_row * ROW_BYTES
    o_base = o_base_row * ROW_BYTES

    # V: host-staged verbatim (V has no producer kernel in this repo), same
    # layout attn_v_16x60.setup() would build from a file.
    v_buf = np.zeros((N_CHAN, PV_STRIDE_ROWS * LANES), dtype=np.float32)
    for h in range(N_HEAD):
        for t in range(D):
            v_buf[h * D + t, :N_TOK] = V[h, t, :]
    state.xmem.write_address(v_base, bytearray(v_buf.astype(np.float32).tobytes()))

    for h in range(N_HEAD):
        lo = h * D
        q_path = tmp_path / f"q_shared_h{h}.bin"
        k_path = tmp_path / f"k_shared_h{h}.bin"
        q_path.write_bytes(Q[lo:lo + D].tobytes())
        k_path.write_bytes(K[lo:lo + D].tobytes())

        # IpuApp.run() -> run_test() calls load_program_from_binary() but
        # never resets program_counter/is_halted; re-running any kernel
        # against a state left halted by a prior run needs an explicit reset
        # first (the same idiom test_full_layer_l5_packed.py uses). This is
        # test-harness state management, not App-class logic.
        state.program_counter = 0

        qk_app = QkScores16x60App(
            inst_path=_QK_INST_BIN,
            query_path=q_path,
            key_path=k_path,
            output_path=None,
            # k_base_row is NOT overridden: qk_scores_16x60.asm hardwires
            # K_BASE to cr0, which the ISA hardwires to the constant 0 (see
            # CR_ZERO_REG_INDEX/CR_READ_ONLY_INITIAL_VALUES in ipu_config.py)
            # -- any write to cr0 is silently dropped. K's base row can never
            # be relocated without changing which CR the .asm reads K_BASE
            # from, which is out of scope. K therefore always lands at row 0,
            # so every other region here is placed at row >= K_ROWS (60).
            qrow_base_row=qk_staging_base_row,
            # lands directly in attn_v's P slot for this head
            s_base_row=p_base_row + h * P_HEAD_STRIDE_ROWS,
        )
        _, cycles = qk_app.run(max_cycles=20_000_000, state=state)
        assert cycles > 0

    state.program_counter = 0
    av_app = AttnV16x60App(
        inst_path=_ATTN_V_INST_BIN,
        p_path=None,   # P already in shared XMEM from the qk_scores runs above
        v_path=None,   # V already in shared XMEM, staged above
        output_path=None,
        pbase_row=p_base_row,
        vbase_row=v_base_row,
        obase_row=o_base_row,
    )
    _, cycles = av_app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    raw = state.xmem.read_address(o_base, N_CHAN * O_CHAN_BYTES)
    got = np.frombuffer(bytes(raw), dtype=np.float32).reshape(N_CHAN, LANES)

    S = np.zeros((N_HEAD, N_TOK, N_TOK), dtype=np.float64)
    for h in range(N_HEAD):
        lo = h * D
        S[h] = Q[lo:lo + D].T.astype(np.float64) @ K[lo:lo + D].astype(np.float64)

    expected = np.zeros((N_HEAD, N_TOK, D), dtype=np.float64)
    for h in range(N_HEAD):
        for t in range(D):
            for i in range(N_TOK):
                expected[h, i, t] = np.sum(
                    S[h, i, :] * V[h, t, :].astype(np.float64)
                )

    for h in range(N_HEAD):
        for t in range(D):
            np.testing.assert_allclose(
                got[h * D + t, :N_TOK].astype(np.float64), expected[h, :, t],
                rtol=1e-4, atol=1e-3,
                err_msg=(
                    f"shared-state chain mismatch vs numpy float64 reference "
                    f"(head {h}, channel {t})"
                ),
            )
