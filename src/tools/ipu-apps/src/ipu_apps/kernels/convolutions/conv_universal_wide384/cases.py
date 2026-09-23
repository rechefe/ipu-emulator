"""Runnable cases for conv_universal_wide384: random FP32 data vs a NumPy conv2d."""
from ipu_apps.kernels.convolutions.universal_cases import conv2d_case

CASES = {
    # width=384 (cpr=3), 8 rows: also the .asm's standalone template defaults.
    "default": conv2d_case(defaults=dict(in_channels=1, out_channels=2, height=8,
                                         width=384, seed=0)),
}
