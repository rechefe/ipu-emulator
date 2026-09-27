"""residual_add: every case, the MobileViT-S channel counts, and the
channel bound that keeps the fixed XMEM regions from overlapping."""
from ipu_apps.kernel_registry import kernel_spec
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.elementwise.residual_add.app import MAX_CHANNELS

test_case = case_tests(__package__, sweep=[
    dict(num_channels=96),
    dict(num_channels=128),
    dict(num_channels=160),
])


def test_channel_count_is_bounded_by_the_xmem_regions():
    spec = kernel_spec("residual_add")
    assert spec.check(num_channels=MAX_CHANNELS)
    assert not spec.check(num_channels=MAX_CHANNELS + 1)
    assert not spec.check(num_channels=0)
