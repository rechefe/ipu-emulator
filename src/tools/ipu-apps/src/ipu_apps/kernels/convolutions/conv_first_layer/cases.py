"""Runnable case for conv_first_layer: the fixed 256x256x3 -> 128x128x16
stride-2 conv + bias -> ReLU, vs a NumPy conv2d."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

CASES = {
    # The shape is fixed; the channel/spatial options exist only because the
    # shared case builder takes them -- any other value is refused by SPEC.
    "default": conv2d_case(stride=2, bias=True, defaults=dict(
        in_channels=3, out_channels=16, height=256, width=256, seed=42)),
}
