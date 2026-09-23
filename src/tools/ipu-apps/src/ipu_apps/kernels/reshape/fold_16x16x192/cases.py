"""Runtime cases for fold_16x16x192."""
from ipu_apps.kernels.reshape.fold_16x16x192 import app
from ipu_apps.kernels.reshape.cases import fold_cases

CASES = fold_cases(app, seed=0x018)
