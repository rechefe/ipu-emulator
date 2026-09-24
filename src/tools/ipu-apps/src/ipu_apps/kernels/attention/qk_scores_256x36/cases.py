"""Runtime cases for qk_scores_256x36: S = Q^T K against NumPy, per key group."""
from ipu_apps.kernels.attention.cases import qk_scores_cases
from ipu_apps.kernels.attention.qk_scores_256x36 import app

CASES = qk_scores_cases(app, seed=0x9C0)
