"""Wide-vector FP32 end-to-end test for attn_scores_km_256x36.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
S[i, s] = sum_c Q[i, c] * K[s, c] is computed directly.

Scores are stored KEY-major (each key column contiguous), which is the
distinguishing property of this kernel vs the query-major `qk_scores_256x36`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_scores_km_256x36 import (
    AttnScoresKM256x36App, N_TOK, D, N_TG, N_TPG, N_HEADS, LANES,
)

_INST_BIN = Path(os.environ["ATTN_SCORES_KM_256X36_INST_BIN"])

_HEAD = 1       # exercise a non-zero head so the head-slicing is covered


def test_attn_scores_km_256x36_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xD00)

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
    app = AttnScoresKM256x36App(
        inst_path=_INST_BIN,
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
    # Key-major group order: row (s, g) holds queries [g*N_TPG, (g+1)*N_TPG).
    got = raw.reshape(N_TOK, N_TG, LANES)

    for g in range(N_TG):
        lo_q = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[lo_q : lo_q + N_TPG, :].T,
            rtol=1e-4, atol=1e-3,
            err_msg=f"key-major scores mismatch for query group {g}",
        )
