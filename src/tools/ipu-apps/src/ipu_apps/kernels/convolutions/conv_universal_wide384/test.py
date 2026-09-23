"""conv_universal_wide384: every case, a sweep (more rows/channels and
the width=512 shape, cpr=4), and the odd-out_channels refusal."""
import numpy as np
import pytest

from ipu_apps.kernel_registry import kernel_spec
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.convolutions.conv_universal_wide384.app import ConvUniversalWide384App

test_case = case_tests(__package__, sweep=[
    dict(in_channels=2, out_channels=2, height=16, width=384),
    dict(in_channels=3, out_channels=4, height=8, width=384),
    dict(in_channels=1, out_channels=2, height=4, width=512),   # cpr=4
])


def test_odd_out_channels_rejected(tmp_path):
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(np.zeros((1, 8, 384), dtype=np.float32).tobytes())
    with pytest.raises(ValueError):
        ConvUniversalWide384App(
            inst_path="unused", input_path=input_file,
            kernel=np.zeros((3, 1, 3, 3), dtype=np.float32), output_path=None,
            width=384, rows=8, in_channels=1, out_channels=3,
        )
    assert not kernel_spec("conv_universal_wide384").check(
        in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, dilation=1,
        groups=1, has_bias=False, apply_relu=False, height=8, width=384)
