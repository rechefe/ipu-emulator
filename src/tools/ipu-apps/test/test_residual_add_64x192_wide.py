"""Wide-vector FP32 end-to-end test for residual_add_64x192 (Layer 4).

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
result is computed directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.residual_add_64x192 import (
    ResidualAdd64x192App, N_ROWS, N_TOK, LANES, OUTPUT_ROW_BYTES,
)

_INST_BIN = Path(os.environ["RESIDUAL_ADD_64X192_INST_BIN"])


def test_residual_add_64x192_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x64192)

    # One channel per row: 128-lane rows with N_TOK valid tokens each.
    A = np.zeros((N_ROWS, LANES), dtype=np.float32)
    B = np.zeros((N_ROWS, LANES), dtype=np.float32)
    A[:, :N_TOK] = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_TOK))
    B[:, :N_TOK] = rng.uniform(-1.0, 1.0, size=(N_ROWS, N_TOK))

    a_path = tmp_path / "a_fp32.bin"
    b_path = tmp_path / "b_fp32.bin"
    a_path.write_bytes(A.tobytes())
    b_path.write_bytes(B.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = ResidualAdd64x192App(
        inst_path=_INST_BIN,
        input_a_path=a_path,
        input_b_path=b_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = (A + B)[:, :N_TOK]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_ROWS * (OUTPUT_ROW_BYTES // 4), (
        f"output has {raw.size} floats, expected {N_ROWS * (OUTPUT_ROW_BYTES // 4)}"
    )
    # Crop each whole output row down to the valid token count.
    got = raw.reshape(N_ROWS, OUTPUT_ROW_BYTES // 4)[:, :N_TOK]

    np.testing.assert_allclose(
        got, expected, rtol=1e-4, atol=1e-3,
        err_msg="residual add output does not match A + B",
    )
