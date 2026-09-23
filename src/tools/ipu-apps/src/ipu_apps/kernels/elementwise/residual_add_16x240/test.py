"""residual_add_16x240: every case in cases.py, plus the one-channel-per-row contract."""
import numpy as np

from ipu_apps.kernel_registry.cases import run_case
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.elementwise.residual_add_16x240.app import LANES, N_CH, N_TOK, OUTPUT_ROW_BYTES
from ipu_apps.kernels.elementwise.residual_add_16x240.cases import CASES

test_case = case_tests(__package__)


def test_one_channel_per_row(tmp_path):
    """Pin the row contract: each channel owns a WHOLE 512-byte row.

    N_TOK=16 puts only 64 bytes of payload in a 512-byte row. Packing several
    channels per row at a 64-byte stride would be a bug; this test reads the
    uncropped rows the harness dumps beside its output and asserts the valid
    prefix is the result (checked by the case) while the rest of the row stays
    zero padding.
    """
    run_case("residual_add_16x240", CASES["default"], workspace=tmp_path)

    rows = np.fromfile(tmp_path / "output.rows.bin", dtype="<f4")
    assert rows.size == N_CH * LANES, (
        f"raw rows have {rows.size} floats, expected {N_CH * LANES} "
        "(one whole row per channel)"
    )
    rows = rows.reshape(N_CH, LANES)
    assert OUTPUT_ROW_BYTES == LANES * 4

    # The cropped output file is the valid token prefix of the raw rows.
    cropped = np.fromfile(tmp_path / "output.bin", dtype="<f4").reshape(N_CH, N_TOK)
    np.testing.assert_array_equal(rows[:, :N_TOK], cropped)

    # Padding lanes: inputs are zero-padded and the op is an add, so the tail
    # of every output row is exactly zero. A nonzero tail would mean a second
    # channel's data had been packed into this row.
    tail = rows[:, N_TOK:]
    np.testing.assert_array_equal(
        tail, np.zeros_like(tail),
        err_msg="row tail is not zero -- channels may be sharing a row",
    )
