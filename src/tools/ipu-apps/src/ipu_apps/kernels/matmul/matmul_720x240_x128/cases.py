"""Runtime cases for matmul_720x240_x128."""
from ipu_apps.kernels.matmul.matmul_720x240_x128 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
