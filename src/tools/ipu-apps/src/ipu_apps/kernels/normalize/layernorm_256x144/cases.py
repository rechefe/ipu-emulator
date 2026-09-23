"""Runtime cases for layernorm_256x144 (normally-distributed inputs)."""
from ipu_apps.kernels.normalize.layernorm_256x144 import app
from ipu_apps.kernels.normalize.cases import layernorm_cases, randn_inputs

CASES = layernorm_cases(app, inputs=randn_inputs, seed=42, rtol=1e-4, atol=1e-4)
