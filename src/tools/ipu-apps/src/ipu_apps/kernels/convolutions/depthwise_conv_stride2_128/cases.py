"""Runnable cases for depthwise_conv_stride2_128: random FP32 data vs a NumPy
stride-2 depthwise conv2d."""
from ipu_apps.kernels.convolutions.cases import conv2d_case

CASES = {
    "default": conv2d_case(stride=2, depthwise=True,
                           defaults=dict(channels=2, height=8, width=128, seed=3)),
}
