"""Runnable cases for pointwise_conv_unified_bn_activation: 1x1 conv + bias -> ReLU vs NumPy."""
from ipu_apps.kernels.convolutions.cases import conv2d_case

DEFAULTS = dict(in_channels=16, out_channels=8, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(kernel_size=1, bias=True, defaults=DEFAULTS),
    # Zero weights and a negative bias: ReLU must write exact zeros everywhere.
    "negative_bias": conv2d_case(kernel_size=1, bias=True, defaults=dict(DEFAULTS, in_channels=8),
                                 data="negative_bias"),
}
