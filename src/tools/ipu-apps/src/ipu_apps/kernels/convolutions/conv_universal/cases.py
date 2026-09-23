"""Runnable cases for conv_universal: random FP32 data vs a NumPy conv2d."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

DEFAULTS = dict(in_channels=16, out_channels=4, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(defaults=DEFAULTS),
    # No ReLU: all-negative pre-activation sums must survive, not be clamped.
    "negative_outputs": conv2d_case(defaults=dict(DEFAULTS, in_channels=4), data="negative"),
}
