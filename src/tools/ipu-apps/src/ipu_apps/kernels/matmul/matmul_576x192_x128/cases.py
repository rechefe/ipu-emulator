"""Runtime cases for matmul_576x192_x128."""
from ipu_apps.kernels.matmul.matmul_576x192_x128 import app
from ipu_apps.kernels.matmul.cases import matmul_cases

CASES = matmul_cases(app)
