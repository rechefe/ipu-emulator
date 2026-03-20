"""End-to-end regression tests for the 16x16 256->128 channel convolution app.

Assemble -> load -> run -> compare output against golden reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipu_apps.conv_16x16_256to128 import Conv16x16x256to128App


_CONV_INST_BIN = Path(os.environ["CONV16X16_256TO128_INST_BIN"])
_CONV_DATA_DIR = Path(os.environ["CONV16X16_256TO128_DATA_DIR"])


def _run_conv(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    """Run 16x16 256->128 conv app for a given dtype, return (output_bytes, cycles)."""
    data_dir = _CONV_DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")

    input_file = data_dir / f"input_{dtype_dir}.bin"
    kernel_file = data_dir / f"kernel_{dtype_dir}.bin"
    if not input_file.exists() or not kernel_file.exists():
        pytest.skip(f"Missing data files in {data_dir}")

    output = tmp_path / "output.bin"
    app = Conv16x16x256to128App(
        inst_path=_CONV_INST_BIN,
        input_path=input_file,
        kernel_path=kernel_file,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=200_000_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8", "INT8", "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "FP8_E4M3", "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "FP8_E5M2", "out_fp8_e5m2_acc_fp32.bin"),
])
def test_conv_16x16_256to128(tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str) -> None:
    actual, cycles = _run_conv(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _CONV_DATA_DIR / dtype_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
