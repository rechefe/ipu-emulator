"""attn_v_16x60: every case in cases.py, plus padding isolation.

AGG.SUM.FIRST reduces across lanes, so the trailing 112 padding lanes of every
P and V row must be excluded by valid_elements, not by relying on the
harness's zero-fill.
"""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.attn_v_16x60.app import LANES, N_CHAN, N_TOK
from ipu_apps.kernels.attention.cases import padded_runs

test_case = case_tests(__package__)


def test_padding_is_inert(tmp_path):
    """Refill the padding lanes with garbage (a chained producer would leave
    another stream's data there) and assert the valid-lane output is
    bit-identical to the all-zero-padding run."""
    zero_padded, garbage_padded = (
        raw.reshape(N_CHAN, LANES)[:, :N_TOK]
        for raw in padded_runs("attn_v_16x60", tmp_path, (0.0, 1e3))
    )
    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_16x60 output changed when padding lanes were filled with "
            "garbage -- AGG reduction is not structurally isolated from "
            "unused lanes"
        ),
    )
