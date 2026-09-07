"""Wide-vector FP32 end-to-end test for matmul_432x144_x128.

Runs the REAL kernel binary against a numpy reference in wide-vector debug mode.
No checked-in golden: FP32 inputs are generated here and the expected result is
computed directly, so the test does not depend on the quantized INT8/FP8 path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.matmuls.matmul_432x144_x128 import (
    MatMul432x144x128App, K, N_OUT, N_TG, N_TOK, LANES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/matmuls/matmul_432x144_x128/matmul_432x144_x128.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "matmul_432x144_x128.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_matmul_432x144_x128_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xC0FFEE)
    D = rng.uniform(-1.0, 1.0, size=(K, N_TG * N_TOK)).astype(np.float32)
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
    app = MatMul432x144x128App(
        inst_path=inst_file,
        input_path=data_path,
        weights_path=weights_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    expected = W @ D                      # C[j, t] = sum_k W[j,k] * D[k,t]

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
