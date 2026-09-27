"""Runnable cases for pointwise_conv_unified: random FP32 data vs a NumPy 1x1 conv2d."""
from ipu_apps.kernels.convolutions.cases import conv2d_case

DEFAULTS = dict(in_channels=16, out_channels=8, height=16, width=16, seed=0)

CASES = {
    "default": conv2d_case(kernel_size=1, bias=False, defaults=DEFAULTS),
}
