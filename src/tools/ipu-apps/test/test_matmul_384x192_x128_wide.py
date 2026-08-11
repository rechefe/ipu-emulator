"""Wide-vector FP32 end-to-end test for matmul_384x192_x128.

Runs the REAL kernel binary against a numpy reference in wide-vector debug mode.
No checked-in golden: FP32 inputs are generated here and the expected result is
computed directly, so the test does not depend on the quantized INT8/FP8 path.

Single token group (N_TOK=64 <= LANES): one store per output channel. In wide
mode that store fills a whole 512 B row, so the output is one row per channel
with the first N_TOK lanes valid -- the narrow harness's packed/overlapping
store layout has no wide equivalent.

This is the FFN1 (expansion) matmul: the store applies `silu` (x*sigmoid(x)),
the FFN nonlinearity, rather than `identity` -- FFN1 and FFN2 stacked with no
activation between them collapse to a single linear layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmul_384x192_x128 import (
    MatMul384x192x128App, K, N_OUT, N_TOK, LANES,
)

_INST_BIN = Path(os.environ["MATMUL_384X192_X128_INST_BIN"])


def test_matmul_384x192_x128_wide_fp32(tmp_path: Path) -> None:
    rng = np.random.RandomState(0xC0FFEE)
    D = rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    data_path = tmp_path / "input_fp32.bin"
    weights_path = tmp_path / "weights_fp32.bin"
    data_path.write_bytes(D.tobytes())
    weights_path.write_bytes(W.tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = MatMul384x192x128App(
        inst_path=_INST_BIN,
        input_path=data_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    pre_act = W @ D                       # C[j, t] = sum_k W[j,k] * D[k,t]
    expected = pre_act * (1.0 / (1.0 + np.exp(-pre_act)))  # silu = x * sigmoid(x)

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_OUT * LANES, (
        f"output has {raw.size} floats, expected {N_OUT * LANES}"
    )
    got = raw.reshape(N_OUT, LANES)

    np.testing.assert_allclose(
        got[:, :N_TOK], expected,
        rtol=1e-4, atol=1e-3,
        err_msg="matmul output mismatch",
    )
