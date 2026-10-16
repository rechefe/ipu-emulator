"""Runtime cases for matmul_64x64x64."""

from ipu_apps.kernels.matmul.cases import matmul_cases
from ipu_apps.kernels.matmul.matmul_64x64x64 import app

CASES = matmul_cases(app)
