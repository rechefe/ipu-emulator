"""Runtime cases for unfold_32x32x144."""

from ipu_apps.kernels.reshape.cases import unfold_cases
from ipu_apps.kernels.reshape.unfold_32x32x144 import app

CASES = unfold_cases(app, seed=0x032)
