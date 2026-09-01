"""Wide-vector FP32 end-to-end test for proj_outproj_192_p4.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
result is computed directly per stream, exactly like every existing
matmul_*_x128 test in this directory.

Multi-stream (P=4): one shared weight matrix W applied independently to 4
per-stream activation blocks D[p], C[p] = W @ D[p]. Single token group
(N_TOK=64 <= LANES): one store per output channel per stream, each filling
a whole 512 B row with the first N_TOK lanes valid.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.projections.proj_outproj_192_p4 import (
    ProjOutProj192P4App, K, N_OUT, N_TOK, N_STREAM, LANES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/projections/proj_outproj_192_p4/proj_outproj_192_p4.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proj_outproj_192_p4.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_proj_outproj_192_p4_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xC0FFEE)
    D = [rng.uniform(-1.0, 1.0, size=(K, N_TOK)).astype(np.float32) for _ in range(N_STREAM)]
    W = rng.uniform(-1.0, 1.0, size=(N_OUT, K)).astype(np.float32)

    input_paths = []
    for p in range(N_STREAM):
        data_path = tmp_path / f"input_fp32_p{p}.bin"
        data_path.write_bytes(D[p].tobytes())
        input_paths.append(data_path)

    weights_path = tmp_path / "weights_fp32.bin"
    weights_path.write_bytes(W.tobytes())

    output_paths = [tmp_path / f"output_p{p}.bin" for p in range(N_STREAM)]

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = ProjOutProj192P4App(
        inst_path=inst_file,
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
    )
    state, cycles = app.run(max_cycles=80_000_000, state=state)
    assert cycles > 0

    for p in range(N_STREAM):
        expected = W @ D[p]                   # C[p][j, t] = sum_k W[j,k] * D[p][k,t]

        raw = np.frombuffer(output_paths[p].read_bytes(), dtype=np.float32)
        assert raw.size == N_OUT * LANES, (
            f"stream {p}: output has {raw.size} floats, expected {N_OUT * LANES}"
        )
        got = raw.reshape(N_OUT, LANES)

        np.testing.assert_allclose(
            got[:, :N_TOK], expected,
            rtol=1e-4, atol=1e-3,
            err_msg=f"proj_outproj_192_p4 stream {p} output mismatch",
        )
