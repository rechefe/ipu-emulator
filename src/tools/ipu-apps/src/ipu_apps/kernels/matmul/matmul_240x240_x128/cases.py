"""Runtime cases for matmul_240x240_x128."""
from ipu_apps.kernels.matmul.matmul_240x240_x128 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
