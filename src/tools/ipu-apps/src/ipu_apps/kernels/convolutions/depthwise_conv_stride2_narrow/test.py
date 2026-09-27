"""depthwise_conv_stride2_narrow: every case, a sweep over each packed
width (rows_per_chunk 2/4/8), and its constructor refusals."""
import numpy as np
import pytest

from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.convolutions.depthwise_conv_stride2_narrow.app import (
    DepthwiseConvStride2NarrowApp,
)

test_case = case_tests(__package__, sweep=[
    dict(channels=2, height=8, width=64, seed=3),    # rows_per_chunk=2, minimal
    dict(channels=3, height=32, width=16, seed=11),  # rows_per_chunk=8, odd channel count
])


@pytest.mark.parametrize("rows,cols", [
    (5, 64),     # odd rows
    (16, 16),    # cols=16 -> rows_per_chunk=8 -> needs rows % 32 == 0
    (8, 128),    # cols outside {16, 32, 64}
])
def test_rejects_unsupported_shapes(tmp_path, rows, cols):
    input_file = tmp_path / "input.bin"
    input_file.write_bytes(np.zeros((1, rows, cols), dtype=np.float32).tobytes())
    with pytest.raises(ValueError):
        DepthwiseConvStride2NarrowApp(
            inst_path="unused", input_path=input_file,
            kernel=np.zeros((1, 3, 3), dtype=np.float32),
            output_path=None, rows=rows, cols=cols, channels=1,
        )
