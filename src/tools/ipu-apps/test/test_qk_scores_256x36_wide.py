"""Wide-vector FP32 end-to-end test for qk_scores_256x36.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and S = Q @ K.T is
computed directly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.qk_scores_256x36 import (
    QkScores256x36App, N, D, N_TG, N_TPG, LANES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/qk_scores_256x36/qk_scores_256x36.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qk_scores_256x36.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_qk_scores_256x36_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0x9C0)

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
    app = QkScores256x36App(
        inst_path=inst_file,
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
    # Query-major group order: row (i, g) holds keys [g*N_TPG, (g+1)*N_TPG).
    got = raw.reshape(N, N_TG, LANES)

    for g in range(N_TG):
        lo = g * N_TPG
        np.testing.assert_allclose(
            got[:, g, :N_TPG], expected[:, lo:lo + N_TPG],
            rtol=1e-4, atol=1e-3,
            err_msg=f"QK^T scores mismatch for key group {g}",
        )
