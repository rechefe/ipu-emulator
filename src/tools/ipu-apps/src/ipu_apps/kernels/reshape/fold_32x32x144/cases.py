"""Runtime cases for fold_32x32x144."""
from ipu_apps.kernels.reshape.fold_32x32x144 import app
from ipu_apps.kernels.reshape.unfold_cases import fold_cases

CASES = fold_cases(app, seed=0x034)
