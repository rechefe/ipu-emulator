"""Runtime cases for concat_16x16x128x128."""
from ipu_apps.kernels.reshape.concat_16x16x128x128 import app
from ipu_apps.kernels.reshape.concat_cases import concat_cases

CASES = concat_cases(app)
