"""Runtime cases for residual_add_256x144 (L3): 288 full 128-lane rows, no padding."""
from ipu_apps.kernels.elementwise.residual_add_256x144 import app
from ipu_apps.kernels.elementwise.cases import residual_add_cases

CASES = residual_add_cases(app, seed=0xADD, valid_lanes=app.LANES, rtol=1e-5, atol=1e-5)
