"""Runtime cases for layernorm_16x240 (Layer 5), including degenerate-pass guards."""
from ipu_apps.kernels.normalize.layernorm_16x240 import app
from ipu_apps.kernels.normalize.layernorm_cases import layernorm_cases, uniform_inputs

CASES = layernorm_cases(app, inputs=uniform_inputs, seed=0x16240, rtol=1e-4, atol=1e-3,
                        cropped_output=True, guards=True)
