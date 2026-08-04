"""Wide-vector FP32 end-to-end test for qk_scores_64x48.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and
S[b] = Q[b].T @ K[b] is computed directly for each of the 16 (stream, head)
blocks of Layer 4.

This is the QUERY-major score kernel; it pairs with ``attn_v_64x48``. The
key-major variant ``attn_scores_km_64x48`` produces bit-different results by
design and has its own golden -- the two must never share one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_64x48 import (
    QkScores64x48App, N, D, N_BLOCK, N_TG, N_TPG,
)

_INST_BIN = Path(os.environ["QK_SCORES_64X48_INST_BIN"])


def test_qk_scores_64x48_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x4C0)

    # Inputs are channel-major per block: [block b, token t, channel c]
    # at element (b*D + c)*N + t.
    Q = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(N_BLOCK, D, N)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = QkScores64x48App(
        inst_path=_INST_BIN,
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    # S[b, i, s] = sum_c Q[b, c, i] * K[b, c, s]
    expected = np.einsum("bci,bcs->bis", Q, K)      # [N_BLOCK, N, N]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_BLOCK * N * N_TG * N_TPG, (
        f"output has {raw.size} floats, expected {N_BLOCK * N * N_TG * N_TPG}"
    )
    got = raw.reshape(N_BLOCK, N, N_TPG)            # [block, query, key]

    max_err = float(np.max(np.abs(got - expected)))
    print(f"qk_scores_64x48 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg="QK^T query-major score mismatch",
    )
