"""attn_v_bcast_60: every case in cases.py, plus padding isolation."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.attn_v_bcast_60.app import LANES, N_CHAN, N_TOK
from ipu_apps.kernels.attention.cases import padded_runs

test_case = case_tests(__package__)


def test_padding_is_inert(tmp_path):
    """No AGG: ACC.ADD accumulates each of the 128 lanes (queries)
    independently, so the 112 padding lanes of P and V can only ever waste
    lanes, never contaminate the 16 valid ones. Prove it: refill the padding
    with garbage and assert the valid-lane output is bit-identical.
    """
    zero_padded, garbage_padded = (
        raw.reshape(N_CHAN, LANES)[:, :N_TOK]
        for raw in padded_runs("attn_v_bcast_60", tmp_path, (0.0, 1e3))
    )
    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_bcast_60 output changed when padding lanes were filled "
            "with garbage -- per-lane ACC is not isolated from unused lanes"
        ),
    )
