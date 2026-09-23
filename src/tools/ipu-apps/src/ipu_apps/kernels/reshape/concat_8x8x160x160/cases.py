"""Runtime cases for concat_8x8x160x160."""
from ipu_apps.kernels.reshape.concat_8x8x160x160 import app
from ipu_apps.kernels.reshape.cases import concat_cases

CASES = concat_cases(app)
