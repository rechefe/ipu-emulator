"""End-to-end regression tests for the standard 16->16 channel conv app (8x8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.conv_8x8_16to16 import Conv8x8_16to16App


_INST_BIN = Path(os.environ["CONV8X8_16TO16_INST_BIN"])
_DATA_DIR = Path(os.environ["CONV8X8_16TO16_DATA_DIR"])


def _run(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    input_file = data_dir / f"input_{dtype_dir}.bin"
    kernel_file = data_dir / f"kernel_{dtype_dir}.bin"
    if not input_file.exists() or not kernel_file.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "output.bin"
    app = Conv8x8_16to16App(
        inst_path=_INST_BIN,
        input_path=input_file,
        kernel_path=kernel_file,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=10_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8", "INT8", "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "FP8_E4M3", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "FP8_E5M2", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_conv_8x8_16to16(tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str) -> None:
    actual, cycles = _run(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _DATA_DIR / dtype_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
