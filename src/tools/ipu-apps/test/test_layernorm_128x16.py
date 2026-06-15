"""End-to-end test for the layernorm_128x16 application (wide-vector FP32 mode)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.layernorm_128x16 import LayerNorm128x16App
from ipu_apps.layernorm_128x16.gen_test_data import reference_layernorm

_INST_BIN = Path(os.environ["LAYERNORM_128X16_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYERNORM_128X16_DATA_DIR"])

N_CH  = 16
N_TPG = 128
ROW_BYTES = 512


def _load_fp32_rows(path: Path, n_rows: int) -> np.ndarray:
    """Load n_rows × 512-byte rows as [n_rows, 128] float32."""
    data = np.frombuffer(path.read_bytes(), dtype=np.float32)
    return data.reshape(n_rows, 128)


def test_layernorm_128x16_wide_fp32(tmp_path: Path) -> None:
    data_dir = _DATA_DIR / "wide_fp32"
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    input_path = data_dir / "input_x_fp32.bin"
    gamma_path = data_dir / "gamma_fp32.bin"
    beta_path  = data_dir / "beta_fp32.bin"
    for p in (input_path, gamma_path, beta_path):
        if not p.exists():
            pytest.skip(f"Missing: {p}")

    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = LayerNorm128x16App(
        inst_path=_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=500_000, state=state)
    assert cycles > 0

    # Load inputs for reference
    x     = _load_fp32_rows(input_path, N_CH)[:, :N_TPG]
    gamma = _load_fp32_rows(gamma_path, 1).reshape(-1)[:N_CH]
    beta  = _load_fp32_rows(beta_path,  1).reshape(-1)[:N_CH]

    expected, _, _, _ = reference_layernorm(x, gamma, beta)  # [N_CH, N_TPG]

    actual_raw = _load_fp32_rows(output_path, N_CH)[:, :N_TPG]

    np.testing.assert_allclose(
        actual_raw, expected,
        atol=1e-4, rtol=1e-4,
        err_msg="LayerNorm output does not match reference",
    )
