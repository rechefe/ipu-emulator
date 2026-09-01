"""Wide-vector FP32 end-to-end test for proj_outproj_144_p4.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
result is computed directly per stream, exactly like every existing
matmul_*_x128 / proj_*_p4 test in this directory.

Multi-stream (P=4): one shared weight matrix W applied independently to 4
per-stream activation blocks D[p], C[p] = W @ D[p] per token group.
L3-specific: N=256 tokens per stream = 2 token groups (N_TG=2) of 128, so
D[p] and the expected result are shaped [N_TG, K/N_OUT, 128] and compared
per (stream, tg).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.projections.proj_outproj_144_p4 import (
    ProjOutproj144P4App, K, N_OUT, N_TG, N_TOK, N_STREAM, LANES,
)

ASM_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/ipu_apps/projections/proj_outproj_144_p4/proj_outproj_144_p4.asm"
)


@pytest.fixture(scope="module")
def inst_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proj_outproj_144_p4.bin"
        assemble_to_bin_file(ASM_PATH.read_text(encoding="utf-8"), str(path))
        yield path


def test_proj_outproj_144_p4_wide_fp32(inst_file: Path, tmp_path: Path) -> None:
    rng = np.random.RandomState(0xC0FFEE)
    D = [rng.uniform(-1.0, 1.0, size=(N_TG, K, N_TOK)).astype(np.float32) for _ in range(N_STREAM)]
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
    app = ProjOutproj144P4App(
        inst_path=inst_file,
        input_paths=input_paths,
        weights_path=weights_path,
        output_paths=output_paths,
    )
    state, cycles = app.run(max_cycles=20_000_000, state=state)
    assert cycles > 0

    for p in range(N_STREAM):
        raw = np.frombuffer(output_paths[p].read_bytes(), dtype=np.float32)
        assert raw.size == N_TG * N_OUT * LANES, (
            f"stream {p}: output has {raw.size} floats, expected {N_TG * N_OUT * LANES}"
        )
        got = raw.reshape(N_TG, N_OUT, LANES)

        for tg in range(N_TG):
            expected = W @ D[p][tg]            # C[p][tg][j, t] = sum_k W[j,k] * D[p][tg][k,t]
            np.testing.assert_allclose(
                got[tg][:, :N_TOK], expected,
                rtol=1e-4, atol=1e-3,
                err_msg=f"proj_outproj_144_p4 stream {p} tg {tg} output mismatch",
            )
