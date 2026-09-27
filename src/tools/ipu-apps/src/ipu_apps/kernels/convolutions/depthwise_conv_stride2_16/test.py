"""depthwise_conv_stride2_16: every case, plus a sweep (two channel
pairs, and MobileViT-S's 320-channel stage-5 shape), and the odd-channels
refusal (channels are packed in pairs)."""
import numpy as np
import pytest

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.convolutions.depthwise_conv_stride2_16.app import (
    DepthwiseConvStride2_16App,
)

test_case = case_tests(__package__, sweep=[
    dict(channels=4, seed=7),
    dict(channels=320, seed=13),
])


def test_rejects_odd_channels(tmp_path):
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(np.zeros((3, 16, 16), dtype=np.float32).tobytes())
    with pytest.raises(ValueError):
        DepthwiseConvStride2_16App(
            inst_path="unused", input_path=input_file,
            kernel=np.zeros((3, 3, 3), dtype=np.float32),
            output_path=None, channels=3,
        )
