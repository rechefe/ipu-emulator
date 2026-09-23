"""attn_v_64x48: every case in cases.py, plus padding isolation.

AGG.SUM.FIRST reduces across lanes, so padding must be excluded by
valid_elements rather than by the harness happening to zero-fill it.
"""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.cases import padded_runs

test_case = case_tests(__package__)


def test_padding_is_inert(tmp_path):
    """Refill the trailing 64 (unused) lanes of every P and V row with garbage
    (a real producer in a chained pipeline would leave another stream's data
    there, not zeros) and assert the valid-lane output is bit-identical to the
    all-zero-padding run. This proves the result does not depend on what is in
    the padding lanes."""
    zero_padded, garbage_padded = padded_runs("attn_v_64x48", tmp_path, (0.0, 1e3))
    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_64x48 output changed when padding lanes were filled with "
            "garbage -- AGG reduction is not structurally isolated from "
            "unused lanes"
        ),
    )
