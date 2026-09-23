"""Runtime cases for concat_32x32x96x96."""
from ipu_apps.kernels.reshape.concat_32x32x96x96 import app
from ipu_apps.kernels.reshape.concat_cases import concat_cases

CASES = concat_cases(app)
