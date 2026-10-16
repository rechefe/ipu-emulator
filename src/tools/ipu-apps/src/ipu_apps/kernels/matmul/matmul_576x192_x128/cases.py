"""Runtime cases for matmul_576x192_x128."""

from ipu_apps.kernels.matmul.cases import matmul_cases
from ipu_apps.kernels.matmul.matmul_576x192_x128 import app

CASES = matmul_cases(app)
