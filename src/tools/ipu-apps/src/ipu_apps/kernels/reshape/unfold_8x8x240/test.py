"""unfold_8x8x240: every case in cases.py, plus the output-row contract."""
import numpy as np
import pytest

from ipu_apps.kernel_registry.cases import assemble_kernel, load_cases, run_case
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.reshape.unfold_8x8x240.app import (
    C, H, W, LANES, N_OUT, N_STREAMS, N_TOK, OUTPUT_ROW_BYTES,
)
from ipu_apps.kernels.reshape.cases import spatial_tensor, stream_rows

KERNEL = "unfold_8x8x240"

test_case = case_tests(__package__)


@pytest.fixture(scope="module")
def inst_file(tmp_path_factory):
    return assemble_kernel(KERNEL, tmp_path_factory.mktemp(KERNEL))


def _run(case, inst_file, workspace):
    run_case(KERNEL, load_cases(KERNEL)[case], workspace=workspace, inst_path=inst_file)
    return workspace


def test_output_shape_and_stale_lanes(inst_file, tmp_path):
    """Pin the row contract: one channel per row, 16 valid tokens, stale tail.

    A stream contributes only N_TOK=16 of a row's 128 lanes, and rows are never
    shared between channels. This test reads the UNCROPPED rows the kernel
    actually stored and asserts both halves of that contract: the valid prefix
    matches the reference, and the stale tail is modelled explicitly rather than
    assumed. If a future change ever packed multiple channels into one row, the
    stride assertion here fails.
    """
    workspace = _run("default", inst_file, tmp_path)
    x = spatial_tensor(load_cases(KERNEL)["default"].defaults["seed"], C, H, W)

    # teardown() writes the raw uncropped rows alongside the cropped output.
    rows = np.fromfile(workspace / "output.rows.bin", dtype="<f4")
    assert rows.size == N_STREAMS * N_OUT * LANES, (
        f"raw rows have {rows.size} floats, expected {N_STREAMS * N_OUT * LANES}"
    )
    rows = rows.reshape(N_STREAMS, N_OUT, LANES)

    # One channel per row: each channel occupies a WHOLE 512-byte row.
    assert OUTPUT_ROW_BYTES == LANES * 4
    assert N_TOK * 4 < OUTPUT_ROW_BYTES, "a row must hold one channel, not several"

    expected = stream_rows(x)
    for s in range(N_STREAMS):
        # Valid prefix: lanes 0..15.
        np.testing.assert_allclose(
            rows[s, :, :N_TOK], expected[s, :, :N_TOK],
            rtol=1e-4, atol=1e-3,
            err_msg=f"stream {s} valid prefix mismatch",
        )

        # Stale tail: lanes 16..127 are NOT part of the contract's payload.
        # Model them explicitly. ACC.STRIDE writes 32 elements into slot 0, so
        # lanes 16..31 receive the decimation of the input's zero padding
        # (lanes 64..127 of the source row, which the default case zeroes) and
        # are therefore exactly 0.0. Lanes 32..127 of R_ACC are never written
        # by this kernel and stay 0.0 from reset.
        stale = rows[s, :, N_TOK:]
        np.testing.assert_array_equal(
            stale, np.zeros_like(stale),
            err_msg=(
                f"stream {s}: stale lanes are not the modelled zeros -- the "
                "input padding or R_ACC slot usage changed"
            ),
        )


def test_padding_is_inert(inst_file, tmp_path):
    """ACC.STRIDE with elements_in_row=16 views the WHOLE 128-lane row as 8
    view-rows and every stream's vertical selector (on/on_inv) draws from both
    the real-data view-rows (0..3, lanes 0..63) and the padding view-rows
    (4..7, lanes 64..127) -- see the decimation trace in execute_acc_stride.
    The padding contribution lands in R_ACC[16:32], which teardown crops away,
    but that is a fact about where ACC.STRIDE happens to place it, not a
    guarantee -- so prove it rather than assume it.

    The ``garbage_padding`` case refills the padding lanes (64..127 of each
    channel's source row) with garbage instead of zero; its cropped (valid,
    lanes 0..15) output must be bit-identical to the zero-padded run.
    """
    zero_padded = _run("default", inst_file, tmp_path / "zero") / "output.bin"
    garbage_padded = _run("garbage_padding", inst_file, tmp_path / "garbage") / "output.bin"
    assert garbage_padded.read_bytes() == zero_padded.read_bytes(), (
        "unfold_8x8x240 cropped output changed when the source row's padding "
        "lanes (64..127) were filled with garbage -- the padding decimation is "
        "not structurally isolated from the valid output"
    )
