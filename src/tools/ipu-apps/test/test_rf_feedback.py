"""End-to-end tests for the rf_feedback application.

Tests the aaq RF feedback round-trip for all 3 dtypes and all 3 agg post-functions
(value, inv, inv_sqrt).  Also prints diagnostic tables showing the intermediate
values at each stage of the feedback path.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from ipu_apps.rf_feedback import RfFeedbackApp
from ipu_apps.rf_feedback.gen_test_data import compute_reference, _ones_byte
from ipu_emu.ipu_math import DType, ipu_mult


_INST_BIN = Path(os.environ["RF_FEEDBACK_INST_BIN"])
_DATA_DIR = Path(os.environ["RF_FEEDBACK_DATA_DIR"])

_POST_FN_NAMES = ["value", "inv", "inv_sqrt"]

# (dtype_dir, dtype_str, scalar_byte, golden_name)
_PARAMS = [
    ("int8",     "INT8",   0x04,  "out_int8_acc_int32.bin"),
    ("fp8_e4m3", "fp8_e4", 0x40,  "out_fp8_e4m3_acc_fp32.bin"),
    ("fp8_e5m2", "fp8_e5", 0x40,  "out_fp8_e5m2_acc_fp32.bin"),
]


def _print_diagnostics(dtype_name: str, dtype: DType, scalar_byte: int, data_bytes: bytes) -> None:
    golden, diag = compute_reference(scalar_byte, data_bytes, dtype)
    one = _ones_byte(dtype)

    print(f"\n=== RF Feedback Diagnostics ({dtype_name}) ===")
    print(f"Input scalar byte:           0x{scalar_byte:02x}")
    print(f"ones byte:                   0x{one:02x}")
    mult_s = diag['mult_scalar']
    if isinstance(mult_s, float):
        mult_hex = struct.unpack("<I", struct.pack("<f", mult_s))[0]
    else:
        mult_hex = mult_s & 0xFFFFFFFF
    print(f"After ipu_mult(scalar, one): {mult_s}  (hex: 0x{mult_hex:08x})")

    for fn_name, aaq_raw, fb_key in [
        ("value",    diag["aaq0"], "feedback_byte_value"),
        ("inv",      diag["aaq1"], "feedback_byte_inv"),
        ("inv_sqrt", diag["aaq2"], "feedback_byte_inv_sqrt"),
    ]:
        fb_byte = diag[fb_key]
        fmt = "<i" if dtype == DType.INT8 else "<f"
        try:
            aaq_val = struct.unpack(fmt, struct.pack("<I", aaq_raw))[0]
        except Exception:
            aaq_val = "?"
        out0 = ipu_mult(fb_byte, data_bytes[0], dtype)
        print(f"\nagg sum {fn_name}:")
        print(f"  aaq raw (uint32):          0x{aaq_raw:08x}  ({aaq_val})")
        print(f"  aaq & 0xFF (low byte):     0x{fb_byte:02x}")
        print(f"  mult.ve.aaq output[0]:     {out0}")


def _run(
    tmp_path: Path,
    dtype_dir: str,
    dtype_str: str,
) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    scalar_path = data_dir / f"scalar_{dtype_dir}.bin"
    data_path   = data_dir / f"data_{dtype_dir}.bin"
    if not scalar_path.exists() or not data_path.exists():
        pytest.skip(f"Missing data files in {data_dir}")
    output = tmp_path / "output.bin"
    app = RfFeedbackApp(
        inst_path=_INST_BIN,
        scalar_path=scalar_path,
        data_path=data_path,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=10_000)
    return output.read_bytes(), cycles


@pytest.mark.parametrize("dtype_dir,dtype_str,scalar_byte,golden_name", _PARAMS)
def test_rf_feedback(
    tmp_path: Path,
    dtype_dir: str,
    dtype_str: str,
    scalar_byte: int,
    golden_name: str,
) -> None:
    actual, cycles = _run(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0

    data_dir  = _DATA_DIR / dtype_dir
    data_path = data_dir / f"data_{dtype_dir}.bin"
    dtype_map = {"INT8": DType.INT8, "fp8_e4": DType.E4, "fp8_e5": DType.E5}
    dtype     = dtype_map[dtype_str]

    _print_diagnostics(dtype_dir, dtype, scalar_byte, data_path.read_bytes())

    golden = data_dir / golden_name
    if golden.exists():
        assert actual == golden.read_bytes()
