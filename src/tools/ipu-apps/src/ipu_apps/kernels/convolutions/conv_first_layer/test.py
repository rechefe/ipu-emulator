"""conv_first_layer: its one fixed-shape case, plus a refusal of any other shape."""
from ipu_apps.kernel_registry import kernel_spec
from ipu_apps.kernel_registry.testing import case_tests

test_case = case_tests(__package__)


def test_only_the_fixed_shape_is_supported():
    query = dict(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1,
                 dilation=1, groups=1, has_bias=True, apply_relu=True, height=256, width=256)
    spec = kernel_spec("conv_first_layer")
    assert spec.check(**query)
    for change in (dict(height=128), dict(out_channels=8), dict(stride=1), dict(apply_relu=False)):
        assert not spec.check(**(query | change)), change
