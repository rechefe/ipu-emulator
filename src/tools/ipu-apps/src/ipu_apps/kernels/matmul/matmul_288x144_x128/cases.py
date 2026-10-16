"""Runtime cases for matmul_288x144_x128."""

from ipu_apps.kernels.matmul.cases import matmul_cases
from ipu_apps.kernels.matmul.matmul_288x144_x128 import app

CASES = matmul_cases(app, activation="silu")
