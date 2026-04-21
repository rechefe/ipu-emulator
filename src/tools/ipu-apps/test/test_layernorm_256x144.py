"""End-to-end regression tests for the layernorm_256x144 application."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.layernorm_256x144 import LayerNorm256x144App


_INST_BIN = Path(os.environ["LAYERNORM_256X144_INST_BIN"])
_DATA_DIR = Path(os.environ["LAYERNORM_256X144_DATA_DIR"])


def _run(tmp_path: Path) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / "fp8_e4m3"
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    input_path = data_dir / "input_fp8_e4m3.bin"
    gamma_path = data_dir / "gamma_fp8_e4m3.bin"
    beta_path  = data_dir / "beta_fp8_e4m3.bin"
    for p in (input_path, gamma_path, beta_path):
        if not p.exists():
            pytest.skip(f"Missing data file: {p}")
    output = tmp_path / "output.bin"
    app = LayerNorm256x144App(
        inst_path=_INST_BIN,
        input_path=input_path,
        gamma_path=gamma_path,
        beta_path=beta_path,
        output_path=output,
    )
    _, cycles = app.run(max_cycles=5_000_000)
    return output.read_bytes(), cycles


def test_layernorm_256x144(tmp_path: Path) -> None:
    actual, cycles = _run(tmp_path)
    assert cycles > 0
    golden = _DATA_DIR / "fp8_e4m3" / "out_fp8_e4m3_acc_int32.bin"
    if golden.exists():
        assert actual == golden.read_bytes()
