"""Runtime cases for matmul_128x64x64."""
from ipu_apps.kernels.matmul.matmul_128x64x64 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
