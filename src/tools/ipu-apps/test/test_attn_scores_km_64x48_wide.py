"""Wide-vector FP32 end-to-end test for attn_scores_km_64x48.

Runs the REAL kernel binary against an inline numpy reference in wide-vector
debug mode. No checked-in golden: FP32 inputs are generated here and
S[p, i, s] = sum_c Q[p, i, c] * K[p, s, c] is computed directly.

Scores are stored KEY-major (each key owns a row holding its 64 query scores),
which is the distinguishing property of this kernel versus the query-major
``qk_scores_64x48``. This kernel is the head of the KEY-MAJOR chain and pairs
only with ``attn_v_bcast_48``; the two chains produce bit-different results by
design and never share a golden.

Results are read back through the kernel's own store path
(ACTIVATE.QUANTIZE + STR_POST_AAQ_REG -> XMEM -> teardown crop), never R_ACC.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attn_scores_km_64x48 import (
    AttnScoresKM64x48App, N, D, P, N_HEAD, N_BLOCK,
)

_INST_BIN = Path(os.environ["ATTN_SCORES_KM_64X48_INST_BIN"])

_HEAD = 2       # exercise a non-zero head so the head-slicing is covered


def test_attn_scores_km_64x48_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xD48)

    # Canonical channel-major files over all (stream, head) blocks: element
    # [stream p, head h, token t, channel c] at ((p*N_HEAD + h)*D + c)*N + t.
    Q = rng.uniform(-1.0, 1.0, size=(P, N_HEAD, D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(P, N_HEAD, D, N)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = AttnScoresKM64x48App(
        inst_path=_INST_BIN,
        input_path=q_path,
        weights_path=k_path,
        output_path=output_path,
        head=_HEAD,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0
    print(f"attn_scores_km_64x48: {cycles} cycles")
    print(state.stats.format_summary())

    # Reference: the selected head, per stream. S[p] = Q[p].T @ K[p] with the
    # channel axis contracted -> [queries, keys], then transposed to key-major.
    q_head = Q[:, _HEAD]                       # [P, D, N]
    k_head = K[:, _HEAD]                       # [P, D, N]
    expected = np.einsum("pci,pcs->psi", q_head, k_head)   # [P, key, query]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == P * N * N, (
        f"output has {raw.size} floats, expected {P * N * N}"
    )
    got = raw.reshape(P, N, N)                 # [stream, key, query]

    max_err = float(np.max(np.abs(got - expected)))
    print(f"attn_scores_km_64x48 max abs error = {max_err:.3e}")

    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg="key-major score mismatch",
    )


def test_attn_scores_km_64x48_head_selection(tmp_path: Path) -> None:
    """A different `head` must score a different block of the same input."""
    rng = np.random.RandomState(0xD49)
    Q = rng.uniform(-1.0, 1.0, size=(P, N_HEAD, D, N)).astype(np.float32)
    K = rng.uniform(-1.0, 1.0, size=(P, N_HEAD, D, N)).astype(np.float32)

    q_path = tmp_path / "q_fp32.bin"
    k_path = tmp_path / "k_fp32.bin"
    q_path.write_bytes(Q.tobytes())
    k_path.write_bytes(K.tobytes())

    for head in (0, N_HEAD - 1):
        output_path = tmp_path / f"output_h{head}.bin"
        state = IpuState(
            wide_vector_debug=True,
            wide_vector_arithmetic=WideVectorArithmetic.FP32,
        )
        app = AttnScoresKM64x48App(
            inst_path=_INST_BIN,
            input_path=q_path,
            weights_path=k_path,
            output_path=output_path,
            head=head,
        )
        app.run(max_cycles=20_000_000, state=state)

        expected = np.einsum(
            "pci,pcs->psi", Q[:, head], K[:, head]
        )
        got = np.frombuffer(
            output_path.read_bytes(), dtype=np.float32
        ).reshape(P, N, N)
        np.testing.assert_allclose(
            got, expected, rtol=1e-4, atol=1e-3,
            err_msg=f"key-major score mismatch for head {head}",
        )
