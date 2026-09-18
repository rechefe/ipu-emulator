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

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.attention.qk_scores_64x48 import (
    QkScores64x48App, N, D, N_BLOCK, N_TG, N_TPG,
    ELEM_BYTES, LANES, ROW_BYTES, S_BASE, ACC_STORE_ROWS,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/attention/qk_scores_64x48/qk_scores_64x48.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qk_scores_64x48.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_qk_scores_64x48_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
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
        inst_path=inst_file,
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


def test_qk_scores_64x48_wide_fp32_padding_lanes_not_stored(inst_file: Path, tmp_path: Path) -> None:
    """valid_elements=N gates both MULT.RC.VE's mask and ACTIVATE.QUANTIZE's
    store window (no AGG here -- lanes are independent, so a garbage-padding
    probe on the VALID lanes proves nothing: garbage in lanes N:128 stays in
    lanes N:128 and never reaches lanes 0:N regardless of gating).

    What the gate actually controls is the STORED EXTENT: with it in place,
    ACTIVATE.QUANTIZE hard-zeros post_aaq_reg lanes N:128 before
    STR_POST_AAQ_REG writes the whole 512 B row, so those bytes are zero no
    matter what garbage sits in K's padding lanes. Stage non-zero garbage in
    K's lanes N:128 (a real producer in a chained pipeline would leave
    another stream's data there) and assert the RAW stored row -- read
    directly from XMEM, bypassing teardown's crop -- is zero past byte
    N*ELEM_BYTES.
    """
    rng = np.random.RandomState(0x4C1)

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
        inst_path=inst_file,
        query_path=q_path,
        key_path=k_path,
        output_path=output_path,
    )

    # Stage K normally via setup(), then overwrite the padding lanes (columns
    # N..LANES-1) of every K row with large non-zero garbage before running.
    app.setup(state)
    garbage = np.full(LANES - N, 1e3, dtype=np.float32).tobytes()
    for row in range(N_BLOCK * D):
        addr = row * ROW_BYTES + N * ELEM_BYTES
        state.xmem.write_address(addr, bytearray(garbage))

    from ipu_emu.emulator import run_test
    state, cycles = run_test(
        inst_path=inst_file,
        setup=lambda s: None,   # inputs already staged above
        teardown=lambda s: None,
        max_cycles=20_000_000,
        state=state,
    )
    assert cycles > 0

    stride = ACC_STORE_ROWS * ROW_BYTES
    for r in range(N_BLOCK * N * N_TG):
        row = state.xmem.read_address(S_BASE + r * stride, ROW_BYTES)
        tail = np.frombuffer(bytes(row[N * ELEM_BYTES:]), dtype=np.float32)
        assert np.all(tail == 0.0), (
            f"row {r}: padding lanes N:{LANES} were stored non-zero -- "
            f"valid_elements is not gating the ACTIVATE.QUANTIZE store window"
        )
