"""Runtime cases for matmul_64x64x64."""
from ipu_apps.kernels.matmul.matmul_64x64x64 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
