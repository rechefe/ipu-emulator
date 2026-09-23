"""Runtime cases for unfold_32x32x144."""
from ipu_apps.kernels.reshape.unfold_32x32x144 import app
from ipu_apps.kernels.reshape.unfold_cases import unfold_cases

CASES = unfold_cases(app, seed=0x032)
