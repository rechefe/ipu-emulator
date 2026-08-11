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
    ELEM_BYTES, ROW_BYTES, K_BASE, K_STRIDE_ROWS,
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


def test_qk_scores_16x60_wide_fp32_padding_lanes_not_stored(tmp_path: Path) -> None:
    """valid_elements=N gates both MULT.RC.VE's mask and ACTIVATE.QUANTIZE's
    store window. There is no AGG in this kernel -- lanes are independent, so
    asserting on the VALID lanes (as attn_v_64x48's padding probe does) proves
    nothing here: garbage in lanes N:128 stays in lanes N:128 and never
    reaches lanes 0:N regardless of gating.

    What the gate actually controls is the STORED EXTENT. Stage non-zero
    garbage in K's padding lanes (columns N..LANES-1 of every channel row --
    a real producer in a chained pipeline would leave another stream's data
    there, not zeros) and assert the stored row is zero past byte N*ELEM_BYTES.
    teardown already dumps whole uncropped rows, so no XMEM bypass is needed.
    """
    rng = np.random.RandomState(0x5C1)

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

    app.setup(state)
    garbage = np.full(LANES - N, 1e3, dtype=np.float32).tobytes()
    for c in range(D):
        addr = c * K_STRIDE_ROWS * ROW_BYTES + N * ELEM_BYTES
        state.xmem.write_address(addr, bytearray(garbage))

    from ipu_emu.emulator import run_test
    state, cycles = run_test(
        inst_path=_INST_BIN,
        setup=lambda s: None,
        teardown=app.teardown,
        max_cycles=20_000_000,
        state=state,
    )
    assert cycles > 0

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    got = raw.reshape(N, N_TG, LANES)
    for g in range(N_TG):
        tail = got[:, g, N_TPG:]
        assert np.all(tail == 0.0), (
            f"key group {g}: padding lanes {N_TPG}:{LANES} were stored "
            f"non-zero -- valid_elements is not gating the ACTIVATE.QUANTIZE "
            f"store window"
        )
