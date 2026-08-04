"""Wide-vector FP32 end-to-end test for residual_add_16x240.

Runs the REAL kernel binary against a numpy reference in wide-vector debug
mode. No checked-in golden: FP32 inputs are generated here and the expected
result is computed directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ipu_emu.ipu_state import IpuState, WideVectorArithmetic

from ipu_apps.residual_add_16x240 import (
    ResidualAdd16x240App, N_CH, N_TOK, LANES, OUTPUT_ROW_BYTES,
)
from ipu_apps.residual_add_16x240.gen_debug_data import pack_rows

_INST_BIN = Path(os.environ["RESIDUAL_ADD_16X240_INST_BIN"])


def _run(tmp_path: Path):
    rng = np.random.RandomState(0x5ADD)
    a = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(N_CH, N_TOK)).astype(np.float32)

    a_path = tmp_path / "a_fp32.bin"
    b_path = tmp_path / "b_fp32.bin"
    a_path.write_bytes(pack_rows(a).tobytes())
    b_path.write_bytes(pack_rows(b).tobytes())
    output_path = tmp_path / "output.bin"

    state = IpuState(
        wide_vector_debug=True,
        wide_vector_arithmetic=WideVectorArithmetic.FP32,
    )
    app = ResidualAdd16x240App(
        inst_path=_INST_BIN,
        input_a_path=a_path,
        input_b_path=b_path,
        output_path=output_path,
    )
    state, cycles = app.run(max_cycles=5_000_000, state=state)
    assert cycles > 0
    return a, b, output_path


def test_residual_add_16x240_wide_fp32(tmp_path: Path) -> None:
    a, b, output_path = _run(tmp_path)

    raw = np.frombuffer(output_path.read_bytes(), dtype=np.float32)
    assert raw.size == N_CH * N_TOK, (
        f"output has {raw.size} floats, expected {N_CH * N_TOK}"
    )
    got = raw.reshape(N_CH, N_TOK)

    np.testing.assert_allclose(
        got, a + b, rtol=1e-4, atol=1e-3,
        err_msg="residual add output does not match A + B",
    )


def test_one_channel_per_row(tmp_path: Path) -> None:
    """Pin the row contract: each channel owns a WHOLE 512-byte row.

    N_TOK=16 puts only 64 bytes of payload in a 512-byte row. Packing several
    channels per row at a 64-byte stride would be a bug; this test reads the
    uncropped rows and asserts the valid prefix is the result while the rest of
    the row stays zero padding.
    """
    a, b, output_path = _run(tmp_path)

    rows_path = output_path.with_suffix(".rows.bin")
    rows = np.frombuffer(rows_path.read_bytes(), dtype=np.float32)
    assert rows.size == N_CH * LANES, (
        f"raw rows have {rows.size} floats, expected {N_CH * LANES} "
        "(one whole row per channel)"
    )
    rows = rows.reshape(N_CH, LANES)
    assert OUTPUT_ROW_BYTES == LANES * 4

    np.testing.assert_allclose(
        rows[:, :N_TOK], a + b, rtol=1e-4, atol=1e-3,
        err_msg="valid token prefix mismatch",
    )

    # Padding lanes: inputs are zero-padded and the op is an add, so the tail
    # of every output row is exactly zero. A nonzero tail would mean a second
    # channel's data had been packed into this row.
    tail = rows[:, N_TOK:]
    np.testing.assert_array_equal(
        tail, np.zeros_like(tail),
        err_msg="row tail is not zero -- channels may be sharing a row",
    )
