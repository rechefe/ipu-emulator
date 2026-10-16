"""Runtime cases for matmul_128x64x128."""

from ipu_apps.kernels.matmul.cases import matmul_cases
from ipu_apps.kernels.matmul.matmul_128x64x128 import app

CASES = matmul_cases(app)
