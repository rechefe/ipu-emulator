"""Runnable cases for depthwise_conv_universal: random FP32 data vs a NumPy
depthwise conv2d (groups == channels)."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

DEFAULTS = dict(channels=4, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(depthwise=True, defaults=DEFAULTS),
    # No ReLU: all-negative pre-activation sums must survive, not be clamped.
    "negative_outputs": conv2d_case(depthwise=True, defaults=DEFAULTS, data="negative"),
}
