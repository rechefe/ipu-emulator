"""Runtime cases for unfold_16x16x192."""
from ipu_apps.kernels.reshape.unfold_16x16x192 import app
from ipu_apps.kernels.reshape.cases import unfold_cases

CASES = unfold_cases(app, seed=0x016)
