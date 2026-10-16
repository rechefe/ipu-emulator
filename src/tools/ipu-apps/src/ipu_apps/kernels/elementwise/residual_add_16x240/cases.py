"""Runtime cases for residual_add_16x240 (L5): one channel per row, output cropped."""

from ipu_apps.kernels.elementwise.cases import residual_add_cases
from ipu_apps.kernels.elementwise.residual_add_16x240 import app

CASES = residual_add_cases(
    app, seed=0x5ADD, valid_lanes=app.N_TOK, rtol=1e-4, atol=1e-3, cropped_output=True
)
