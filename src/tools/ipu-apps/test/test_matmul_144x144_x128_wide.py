"""Wide-vector FP32 end-to-end test for matmul_144x144_x128.

Runs the REAL kernel binary against a numpy reference, in wide-vector debug
mode. No checked-in golden is involved: FP32 inputs are generated here and the
expected result is computed directly, so the test is self-sufficient and does
not depend on the INT8/FP8 quantized data path.

This is the primary correctness check. INT8 is an output-stage concern
(ACTIVATE.QUANTIZE at the XMEM write boundary), not a mode the kernel is
written against, so the kernel body is identical in both modes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_144x144_x128 import (
    MatMul144x144x128App, K, N_OUT, N_TG, N_TOK, LANES, W_STRIDE_ROWS,
)

_INST_BIN = Path(os.environ["MATMUL_144X144_X128_INST_BIN"])


def _write_fp32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.astype(np.float32).tobytes())


def test_matmul_144x144_x128_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xC0FFEE)

    # D is channel-major [K, N_TG*N_TOK]; W is output-major [N_OUT, K].
    D = rng.uniform(-1.0, 1.0, size=(K, N_TG * N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    data_path = tmp_path / "input_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    _write_fp32(data_path, D)
    _write_fp32(weights_path, W)

    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = MatMul144x144x128App(
        inst_path=_INST_BIN,
        input_path=data_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    # C[j, t] = sum_k W[j, k] * D[k, t]
    expected = W @ D                       # [N_OUT, N_TG*N_TOK]

    # Output rows are (tg, j): tg-major, each a 512 B / 128-lane FP32 vector.
    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_OUT * N_TG * LANES, (
        f"output has {raw.size} floats, expected {N_OUT * N_TG * LANES}"
    )
    got = raw.reshape(N_TG, N_OUT, LANES)

    for tg in range(N_TG):
        lo = tg * N_TOK
        np.testing.assert_allclose(
            got[tg, :, :N_TOK], expected[:, lo:lo + N_TOK],
            rtol=1e-4, atol=1e-3,
            err_msg=f"matmul output mismatch for token group {tg}",
        )


def test_weights_padding_is_row_aligned() -> None:
    """K=144 is not a multiple of LANES, so W pads to whole rows per channel."""
    assert W_STRIDE_ROWS == 2
    assert W_STRIDE_ROWS * LANES >= K
