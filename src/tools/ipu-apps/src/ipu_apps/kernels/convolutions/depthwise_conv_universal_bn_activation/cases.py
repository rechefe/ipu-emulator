"""Runnable cases for depthwise_conv_universal_bn_activation: depthwise conv +
bias -> ReLU vs NumPy."""
from ipu_apps.kernels.convolutions.cases import conv2d_case

DEFAULTS = dict(channels=4, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(depthwise=True, bias=True, defaults=DEFAULTS),
    # Zero weights and a negative bias: ReLU must write exact zeros everywhere.
    "negative_bias": conv2d_case(depthwise=True, bias=True, defaults=DEFAULTS,
                                 data="negative_bias"),
}
