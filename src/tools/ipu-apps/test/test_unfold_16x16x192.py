"""End-to-end regression tests for the unfold_16x16x192 application (L4)."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from ipu_apps.unfold_16x16x192 import (
    Unfold16x16x192App,
    C,
    N_STREAMS,
    N_TOK,
    OUTPUT_ROW_BYTES,
)


_INST_BIN = Path(os.environ["UNFOLD_16X16X192_INST_BIN"])
_DATA_DIR = Path(os.environ["UNFOLD_16X16X192_DATA_DIR"])


def _run(tmp_path: Path, dtype_dir: str, dtype_str: str) -> tuple[bytes, int]:
    data_dir = _DATA_DIR / dtype_dir
    if not data_dir.exists():
        pytest.skip(f"Test data not found: {data_dir}")
    input_path = data_dir / f"input_{dtype_dir}.bin"
    if not input_path.exists():
        pytest.skip(f"Missing input file: {input_path}")
    output = tmp_path / "output.bin"
    app = Unfold16x16x192App(
        inst_path=_INST_BIN,
        input_path=input_path,
        output_path=output,
        dtype=dtype_str,
    )
    _, cycles = app.run(max_cycles=5_000_000)
    return output.read_bytes(), cycles


# Note: the dtype directory names here match what gen_test_data.py actually
# writes ("fp8_e4"/"fp8_e5"). The L3 test asks for "fp8_e4m3"/"fp8_e5m2", which
# its generator never produces, so its two FP8 cases skip silently.
@pytest.mark.parametrize("dtype_dir,dtype_str,golden_name", [
    ("int8",   "INT8",   "out_int8_acc_int32.bin"),
    ("fp8_e4", "fp8_e4", "out_fp8_e4_acc_fp32.bin"),
    ("fp8_e5", "fp8_e5", "out_fp8_e5_acc_fp32.bin"),
])
def test_unfold_16x16x192(
    tmp_path: Path, dtype_dir: str, dtype_str: str, golden_name: str
) -> None:
    actual, cycles = _run(tmp_path, dtype_dir, dtype_str)
    assert cycles > 0
    golden = _DATA_DIR / dtype_dir / golden_name
    assert golden.exists(), f"golden missing: {golden}"
    assert actual == golden.read_bytes()


def test_output_shape_and_stale_half(tmp_path: Path) -> None:
    """Each row carries 64 valid tokens; the upper half is stale padding.

    Guards the Option-B layout contract: a stream fills only r_acc slots 0 and 1,
    and STR_ACC_REG writes all 512 bytes regardless. Consumers must read only the
    first 256 bytes of each row.
    """
    actual, _ = _run(tmp_path, "int8", "INT8")
    assert len(actual) == N_STREAMS * C * OUTPUT_ROW_BYTES

    # The kernel never writes r_acc lanes 64..127, which are zero from reset.
    for row in range(0, N_STREAMS * C, 37):        # sample rows across all streams
        upper = actual[row * OUTPUT_ROW_BYTES + N_TOK * 4:(row + 1) * OUTPUT_ROW_BYTES]
        vals = struct.unpack(f"<{N_TOK}i", upper)
        assert all(v == 0 for v in vals), f"row {row} upper half not stale-zero"
