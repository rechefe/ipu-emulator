"""Runtime cases for concat_32x32x96x96."""

from ipu_apps.kernels.reshape.cases import concat_cases
from ipu_apps.kernels.reshape.concat_32x32x96x96 import app

CASES = concat_cases(app)
