"""Runtime cases for proj_outproj_192_p4."""
from ipu_apps.kernels.projections.cases import projection_cases
from ipu_apps.kernels.projections.proj_outproj_192_p4 import app

CASES = projection_cases(app, max_cycles=80_000_000)
