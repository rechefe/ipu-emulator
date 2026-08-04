"""Wide-vector FP32 end-to-end test for matmul_128x128.

Runs the REAL kernel binary against a numpy reference, in wide-vector debug
mode. No checked-in golden is involved: FP32 inputs are generated here and the
expected result is computed directly, so the test is self-sufficient and does
not depend on the INT8/FP8 quantized data path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_128x128 import MatMul128x128App, M, K, N

_INST_BIN = Path(os.environ["MATMUL_128X128_INST_BIN"])


def _write_fp32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.astype(np.float32).tobytes())


def test_matmul_128x128_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x128128)

    A = rng.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N, K)).astype(np.float32)

    input_path = tmp_path / "input_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    _write_fp32(input_path, A)
    _write_fp32(weights_path, W)

    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = MatMul128x128App(
        inst_path=_INST_BIN,
        input_path=input_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = A @ W.T                       # C[m, n] = sum_k A[m,k] * W[n,k]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == M * N, f"output has {raw.size} floats, expected {M * N}"
    got = raw.reshape(M, N)

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)
