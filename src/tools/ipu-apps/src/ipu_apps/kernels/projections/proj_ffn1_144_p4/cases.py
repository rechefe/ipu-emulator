"""Runtime cases for proj_ffn1_144_p4."""
from ipu_apps.kernels.projections.cases import projection_cases
from ipu_apps.kernels.projections.proj_ffn1_144_p4 import app

CASES = projection_cases(app, max_cycles=20_000_000)
