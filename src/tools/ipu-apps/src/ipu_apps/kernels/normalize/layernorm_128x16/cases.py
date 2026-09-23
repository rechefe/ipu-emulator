"""Runtime cases for layernorm_128x16 (normally-distributed inputs)."""
from ipu_apps.kernels.normalize.layernorm_128x16 import app
from ipu_apps.kernels.normalize.layernorm_cases import layernorm_cases, randn_inputs

CASES = layernorm_cases(app, inputs=randn_inputs, seed=42, rtol=1e-4, atol=1e-4,
                        pad_params=True, max_cycles=500_000)
