"""Runnable cases for conv_universal_bn_activation: conv + bias -> ReLU vs NumPy."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

DEFAULTS = dict(in_channels=16, out_channels=4, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(bias=True, defaults=DEFAULTS),
    # Zero weights and a negative bias: ReLU must write exact zeros everywhere.
    "negative_bias": conv2d_case(bias=True, defaults=dict(DEFAULTS, in_channels=8, out_channels=2),
                                 data="negative_bias"),
}
