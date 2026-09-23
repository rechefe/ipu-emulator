"""Runnable cases for depthwise_conv_stride2_16: random FP32 data vs a NumPy
stride-2 depthwise conv2d on the fixed 16x16 shape."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

CASES = {
    "default": conv2d_case(stride=2, depthwise=True,
                           defaults=dict(channels=2, height=16, width=16, seed=3)),
}
