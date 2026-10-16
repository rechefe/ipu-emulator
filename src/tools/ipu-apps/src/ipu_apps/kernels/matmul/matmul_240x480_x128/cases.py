"""Runtime cases for matmul_240x480_x128."""

from ipu_apps.kernels.matmul.cases import matmul_cases
from ipu_apps.kernels.matmul.matmul_240x480_x128 import app

CASES = matmul_cases(app)
