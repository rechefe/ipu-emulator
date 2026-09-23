"""depthwise_conv_stride2_128: every case, plus a sweep (the ch_loop
cross-word row advance, an odd channel count, and a large rows*channels that
exercises dynamic region sizing), and the odd-rows refusal."""
import numpy as np
import pytest

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.convolutions.depthwise_conv_stride2_128.app import (
    DepthwiseConvStride2_128App,
)

test_case = case_tests(__package__, sweep=[
    dict(channels=4, height=16, seed=7),
    dict(channels=3, height=32, seed=11),
    dict(channels=32, height=128, seed=13),
])


def test_rejects_odd_rows(tmp_path):
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(np.zeros((1, 5, 128), dtype=np.float32).tobytes())
    with pytest.raises(ValueError):
        DepthwiseConvStride2_128App(
            inst_path="unused", input_path=input_file,
            kernel=np.zeros((1, 3, 3), dtype=np.float32),
            output_path=None, rows=5, channels=1,
        )
