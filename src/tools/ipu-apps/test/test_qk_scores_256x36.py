"""End-to-end regression tests for the qk_scores_256x36 application (Agent C).

Primary check: wide-vector FP32 vs a numpy reference (correctness without quant
noise). Secondary: INT8 and FP8 (E4M3/E5M2) vs the ipu_math goldens.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.qk_scores_256x36 import QkScores256x36App

_INST_BIN = Path(os.environ["QK_SCORES_256X36_INST_BIN"])
_DATA_DIR = Path(os.environ["QK_SCORES_256X36_DATA_DIR"])

N    = 256
D    = 36
N_TG = 2
N_TPG = 128


def test_qk_scores_256x36_wide_fp32(tmp_path: Path) -> None:
    data_dir = _DATA_DIR / "wide_fp32"
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    query_path = data_dir / "query_fp32.bin"
    key_path   = data_dir / "key_fp32.bin"
    for p in (query_path, key_path):
        if not p.exists():
            pytest.skip(f"Missing: {p}")

    output_path = tmp_path / "output.bin"
    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = QkScores256x36App(
        inst_path=_INST_BIN,
        query_path=query_path,
        key_path=key_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0

    # numpy reference straight from the channel-major [D, N] inputs.
    q = np.frombuffer(query_path.read_bytes(), dtype=np.float32).reshape(D, N)
    k = np.frombuffer(key_path.read_bytes(),   dtype=np.float32).reshape(D, N)
    scores = (q.T @ k).astype(np.float32)        # [query i, key s]

    out = np.frombuffer(output_path.read_bytes(), dtype=np.float32).reshape(N * N_TG, 128)
    actual = np.empty((N, N), dtype=np.float32)
    for i in range(N):
        for g in range(N_TG):
            actual[i, g * N_TPG:(g + 1) * N_TPG] = out[i * N_TG + g]

    np.testing.assert_allclose(
        actual, scores, atol=1e-3, rtol=1e-3,
        err_msg="QKᵀ scores do not match numpy reference",
    )


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8",     "INT8",   "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "fp8_e4", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "fp8_e5", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_qk_scores_256x36_dtype(
    tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str
) -> None:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    query_path = data_dir / f"query_{dtype_dir}.bin"
    key_path   = data_dir / f"key_{dtype_dir}.bin"
    if not query_path.exists() or not key_path.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output_path = tmp_path / "output.bin"
    app = QkScores256x36App(
        inst_path=_INST_BIN,
        query_path=query_path,
        key_path=key_path,
        output_path=output_path,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=5_000_000)
    assert cycles > 0

    golden = data_dir / golden_name
    if golden.exists():
        assert output_path.read_bytes() == golden.read_bytes()
