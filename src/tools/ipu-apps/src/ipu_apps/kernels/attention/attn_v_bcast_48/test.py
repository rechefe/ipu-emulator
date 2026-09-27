"""attn_v_bcast_48: every case in cases.py, plus padding isolation."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.cases import padded_runs

test_case = case_tests(__package__)


def test_padding_is_inert(tmp_path):
    """Every lane (query) accumulates independently -- no AGG, no cross-lane
    reduction -- so the trailing 64 padding lanes of P and V can only ever
    waste lanes, never contaminate the 64 valid ones. Prove it: refill the
    padding with garbage (a chained producer would leave real data there, not
    zeros) and assert the valid-lane output is bit-identical.
    """
    zero_padded, garbage_padded = padded_runs("attn_v_bcast_48", tmp_path, (0.0, 1e3))
    np.testing.assert_array_equal(
        garbage_padded, zero_padded,
        err_msg=(
            "attn_v_bcast_48 output changed when padding lanes were filled "
            "with garbage -- per-lane ACC is not isolated from unused lanes"
        ),
    )
