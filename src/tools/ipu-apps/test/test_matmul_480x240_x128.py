"""End-to-end regression tests for the matmul_480x240_x128 application."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.matmul_480x240_x128 import MatMul480x240x128App


_INST_BIN = Path(os.environ["MATMUL_480X240_X128_INST_BIN"])
_DATA_DIR = Path(os.environ["MATMUL_480X240_X128_DATA_DIR"])


def _run(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    input_path = data_dir / f"input_{dtype_dir}.bin"
    weights_path = data_dir / f"weights_{dtype_dir}.bin"
    if not input_path.exists() or not weights_path.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "output.bin"
    app = MatMul480x240x128App(
        inst_path=_INST_BIN,
        input_path=input_path,
        weights_path=weights_path,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=10_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8",     "INT8",     "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "fp8_e4", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "fp8_e5", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_matmul_480x240_x128(
    tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str
) -> None:
    actual, cycles = _run(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _DATA_DIR / dtype_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
