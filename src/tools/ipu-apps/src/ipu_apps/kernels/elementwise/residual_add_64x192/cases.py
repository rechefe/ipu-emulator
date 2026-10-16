"""Runtime cases for residual_add_64x192 (L4): one channel per row, full output rows."""

from ipu_apps.kernels.elementwise.cases import residual_add_cases
from ipu_apps.kernels.elementwise.residual_add_64x192 import app

CASES = residual_add_cases(app, seed=0x64192, valid_lanes=app.N_TOK, rtol=1e-4, atol=1e-3)
