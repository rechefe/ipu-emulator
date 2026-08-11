"""Wide-vector FP32 end-to-end test for matmul_720x240_x128.

Runs the REAL kernel binary against a numpy reference, in wide-vector debug
mode. No checked-in golden is involved: FP32 inputs are generated here and the
expected result is computed directly.

N_TOK=16 < LANES, so each output channel occupies a whole 512 B XMEM row of
which only the first N_TOK*4 bytes are valid. Per kernel_layer_map.md's crop
convention, the PRODUCER emits full, uncropped rows (teardown dumps all
LANES elements per channel) -- this test, as the final consumer, crops to
the valid N_TOK prefix itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_720x240_x128 import MatMul720x240x128App, K, N_OUT, N_TOK, LANES

_INST_BIN = Path(os.environ["MATMUL_720X240_X128_INST_BIN"])


def _write_fp32(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.astype(np.float32).tobytes())


def test_matmul_720x240_x128_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0x720240)

    # D is channel-major [K, N_TOK]; W is output-major [N_OUT, K].
    D = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
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
    # Paths are passed as plain strings: the harness must coerce them itself
    # (a converted __init__ that lost its Path(...) coercion fails here).
    app = MatMul720x240x128App(
        inst_path=_INST_BIN,
        input_path=str(data_path),
        weights_path=str(weights_path),
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    expected = W @ D                       # C[j, t] = sum_k W[j,k] * D[k,t]

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_OUT * LANES, (
        f"output has {raw.size} floats, expected {N_OUT * LANES}"
    )
    got = raw.reshape(N_OUT, LANES)

    np.testing.assert_allclose(got[:, :N_TOK], expected, rtol=1e-4, atol=1e-3)
