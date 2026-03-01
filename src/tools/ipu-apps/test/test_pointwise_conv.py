"""End-to-end regression tests for the pointwise convolution application.

Assemble -> load -> run -> compare output against golden reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.pointwise_conv import PointwiseConvApp


_PCONV_INST_BIN = Path(os.environ["PCONV_INST_BIN"])
_PCONV_DATA_DIR = Path(os.environ["PCONV_DATA_DIR"])


def _run_pointwise_conv(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    """Run pointwise conv app for a given dtype, return (output_bytes, cycles)."""
    data_dir = _PCONV_DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    input_file = data_dir / f"input_{dtype_dir}.bin"
    kernel_file = data_dir / f"kernel_{dtype_dir}.bin"
    if not input_file.exists() or not kernel_file.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "output.bin"
    app = PointwiseConvApp(
        inst_path=_PCONV_INST_BIN,
        input_path=input_file,
        kernel_path=kernel_file,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=2_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8", "INT8", "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "FP8_E4M3", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "FP8_E5M2", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_pointwise_conv(tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str) -> None:
    actual, cycles = _run_pointwise_conv(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _PCONV_DATA_DIR / dtype_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
