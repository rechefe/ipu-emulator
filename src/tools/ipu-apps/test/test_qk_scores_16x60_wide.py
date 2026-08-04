"""Wide-vector FP32 end-to-end test for qk_scores_16x60 (Layer 5).

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and S = Q.T @ K is
computed directly.

This is the QUERY-MAJOR score kernel; it pairs with ``attn_v_16x60``. The
key-major chain (``attn_scores_km_16x60`` + ``attn_v_bcast_60``) computes the
same mathematical scores through a different mapping and has its OWN golden --
the two are bit-different by design and must never share expectations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_16x60 import (
    QkScores16x60App, N, D, N_TG, N_TPG, LANES,
)

_INST_BIN = Path(os.environ["QK_SCORES_16X60_INST_BIN"])


def test_qk_scores_16x60_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x5C0)

    # Inputs are channel-major: element [token t, channel c] at (c*N + t).
    Q = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(D, N)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = QkScores16x60App(
        inst_path=_INST_BIN,
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # S[i, s] = sum_c Q[c, i] * K[c, s]
    expected = Q.T @ K                      # [N, N]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N * N_TG * LANES, (
        f"output has {raw.size} floats, expected {N * N_TG * LANES}"
    )
    # Query-major: one WHOLE row per query, first N_TPG lanes live (one channel
    # per row -- the rest of the row is this query's, just unused).
    got = raw.reshape(N, N_TG, LANES)

    for g in range(N_TG):
        lo = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[:, lo:lo + N_TPG],
            rtol=1e-4, atol=1e-3,
            err_msg=f"QK^T scores mismatch for key group {g}",
        )
