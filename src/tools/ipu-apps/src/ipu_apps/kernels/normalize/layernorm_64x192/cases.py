"""Runtime cases for layernorm_64x192 (Layer 4)."""
from ipu_apps.kernels.normalize.layernorm_64x192 import app
from ipu_apps.kernels.normalize.layernorm_cases import layernorm_cases, uniform_inputs

CASES = layernorm_cases(app, inputs=uniform_inputs, seed=0x64192, rtol=1e-4, atol=1e-3)
