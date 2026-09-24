"""Runtime cases for matmul_432x144_x128."""
from ipu_apps.kernels.matmul.matmul_432x144_x128 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
