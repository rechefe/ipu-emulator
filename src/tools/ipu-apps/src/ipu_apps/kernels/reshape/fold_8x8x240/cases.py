"""Runtime cases for fold_8x8x240."""
from ipu_apps.kernels.reshape.fold_8x8x240 import app
from ipu_apps.kernels.reshape.cases import fold_cases

CASES = fold_cases(app, seed=0x05A)
