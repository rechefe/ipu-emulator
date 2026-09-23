"""softmax_rows: every case, the default case swept over row counts and logit
magnitudes (numerical stability, near-uniform), and degenerate inputs."""
import numpy as np

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.softmax.cases import run_array
from ipu_apps.kernels.softmax.softmax_rows.app import LANES

test_case = case_tests(__package__, sweep=[
    dict(rows=1, scale=3.0, seed=0),
    dict(rows=4, scale=3.0, seed=0),
    dict(rows=8, scale=5.0, seed=1),
    dict(rows=32, scale=5.0, seed=2),
    dict(rows=128, scale=5.0, seed=3),    # one full group (the original target case)
    dict(rows=8, scale=50.0, seed=4),     # numerical-stability: large magnitudes
    dict(rows=8, scale=0.01, seed=5),     # near-uniform logits
    dict(rows=129, scale=3.0, seed=6),    # K>128: 2 groups, last group 1 real row (rest padded)
    dict(rows=200, scale=4.0, seed=7),    # 2 groups, partial last group
    dict(rows=256, scale=3.0, seed=8),    # exactly 2 full groups
    dict(rows=384, scale=5.0, seed=9),    # 3 full groups
    dict(rows=500, scale=3.0, seed=10),   # 4 groups, partial last group
])


def test_constant_row_is_uniform():
    """A row of identical logits must produce the uniform distribution."""
    x = np.full((2, LANES), 3.7, dtype=np.float32)
    _, out = run_array("softmax_rows", x, 1)
    assert np.allclose(out, 1.0 / LANES, atol=1e-6)


def test_argmax_preserved():
    """The largest logit must map to the largest probability per row."""
    x = (np.random.RandomState(7).randn(16, LANES) * 4.0).astype(np.float32)
    _, out = run_array("softmax_rows", x, 1)
    assert np.array_equal(out.argmax(axis=1), x.argmax(axis=1))
