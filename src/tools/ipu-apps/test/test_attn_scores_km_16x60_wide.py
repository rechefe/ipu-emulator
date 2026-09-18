"""Wide-vector FP32 end-to-end test for attn_scores_km_16x60 (Layer 5).

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
S[i, s] = sum_c Q[i, c] * K[s, c] is computed directly.

Scores are stored KEY-major (key s's score column occupies one WHOLE row, one
query per lane), which is the distinguishing property of this kernel vs the
query-major ``qk_scores_16x60``. This is the first half of the KEY-MAJOR chain
(``attn_scores_km_16x60`` -> ``attn_v_bcast_60``); the query-major chain is
bit-different BY DESIGN and the two must never share expectations.

Contraction is ACC.ADD over 60 channels (no AGG), which rounds to float32 at
every step, so a float32 matmul reference is compared at rtol=1e-4/atol=1e-3.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.attn_scores_km_16x60 import (
    AttnScoresKM16x60App, N_TOK, D, N_TG, N_TPG, N_HEADS, LANES,
    ELEM_BYTES, ROW_BYTES, QBASE, Q_CHAN_ROWS,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/attn_scores_km_16x60/attn_scores_km_16x60.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "attn_scores_km_16x60.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


_HEAD = 1       # exercise a non-zero head so the head-slicing is covered


def test_attn_scores_km_16x60_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xD60)

    # Canonical channel-major files: element [token t, channel h*D+c] at
    # (h*D + c)*N_TOK + t, for all N_HEADS heads.
    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnScoresKM16x60App(
        inst_path=inst_file,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=_HEAD,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # This head's channel block, then S[i, s] = sum_c Q[c, i] * K[c, s].
    lo = _HEAD * D
    q_head = Q[lo : lo + D]                 # [D, N_TOK]
    k_head = K[lo : lo + D]                 # [D, N_TOK]
    expected = q_head.T @ k_head            # [N_TOK queries, N_TOK keys]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_TOK * N_TG * LANES, (
        f"output has {raw.size} floats, expected {N_TOK * N_TG * LANES}"
    )
    # Key-major: row s holds the score column S[:, s], query i in lane i.
    # ONE query group at L5, so N_TG == 1 and each key is exactly one row.
    got = raw.reshape(N_TOK, N_TG, LANES)

    for g in range(N_TG):
        lo_q = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[lo_q : lo_q + N_TPG, :].T,
            rtol=1e-4, atol=1e-3,
            err_msg=f"key-major scores mismatch for query group {g}",
        )


def test_attn_scores_km_16x60_wide_fp32_padding_lanes_not_stored(inst_file: Path, tmp_path: Path) -> None:
    """valid_elements=N_TOK gates both MULT.RC.VE's mask (over Q's channel
    columns, the R_CYCLIC operand) and ACTIVATE.QUANTIZE's store window. No
    AGG here -- lanes are independent, so asserting on the VALID lanes (as
    attn_v_64x48's padding probe does) proves nothing: garbage in lanes
    N_TOK:128 stays in lanes N_TOK:128 and never reaches lanes 0:N_TOK
    regardless of gating.

    What the gate actually controls is the STORED EXTENT. Stage non-zero
    garbage in Q's padding lanes (columns N_TOK..LANES-1 of every channel row
    -- a real producer in a chained pipeline would leave another stream's
    data there, not zeros) and assert the stored row is zero past byte
    N_TOK*ELEM_BYTES. teardown already dumps whole uncropped rows.
    """
    rng = np.random.RandomState(0xD61)

    n_chan = N_HEADS * D
    Q = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(n_chan, N_TOK)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnScoresKM16x60App(
        inst_path=inst_file,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=_HEAD,
    )

    app.setup(state)
    garbage = np.full(LANES - N_TOK, 1e3, dtype=np.float32).tobytes()
    for c in range(D):
        addr = QBASE + c * Q_CHAN_ROWS * ROW_BYTES + N_TOK * ELEM_BYTES
        state.xmem.write_address(addr, bytearray(garbage))

    from ipu_emu.emulator import run_test
    state, cycles = run_test(
        inst_path=inst_file,
        setup=lambda s: None,
        teardown=app.teardown,
        max_cycles=20_000_000,
        state=state,
    )
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N_TOK, N_TG, LANES)
    for g in range(N_TG):
        tail = got[:, g, N_TPG:]
        assert np.all(tail == 0.0), (
            f"query group {g}: padding lanes {N_TPG}:{LANES} were stored "
            f"non-zero -- valid_elements is not gating the ACTIVATE.QUANTIZE "
            f"store window"
        )
