"""Runnable cases for depthwise_conv_stride2_narrow: random FP32 data vs a
NumPy stride-2 depthwise conv2d."""
from ipu_apps.kernels.convolutions.cases import conv2d_case

CASES = {
    # cols=32 (rows_per_chunk=4): also the stage-2 .asm's standalone default.
    "default": conv2d_case(stride=2, depthwise=True,
                           defaults=dict(channels=4, height=16, width=32, seed=7)),
}
