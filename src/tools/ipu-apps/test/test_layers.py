"""Tests for the standalone PyTorch nn.Conv2d adapter (convolutions_universal.layers).

Covers: framework-free ``resolve()`` dispatch (mirroring each app's own
SPEC.supports), ``run_layer`` end-to-end against a real
``torch.nn.functional.conv2d`` reference (tolerance-based, FP32
wide-vector mode, since IPU accumulation order differs from PyTorch's), the
``register_layer("Conv2d")`` adapter, and the refusal cases layers.py
documents (non-square params, unsupported groups, stride=2 for plain conv,
bias without apply_relu, width > 128 not covered by any app in this
package).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from ipu_apps.convolutions_universal.layers import (  # noqa: E402
    Conv2dDescription,
    resolve,
    run_layer,
)
from ipu_apps.kernel_registry.layers import UnsupportedLayer, from_layer  # noqa: E402

_TOL = 1e-2


def _reference_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
    *, stride: int, groups: int, relu: bool, padding: int = 1,
) -> torch.Tensor:
    out = F.conv2d(x.unsqueeze(0), weight, bias=bias, padding=padding, stride=stride, groups=groups)[0]
    if relu:
        out = F.relu(out)
    return out


class TestRunLayerStride1:

    def test_plain_conv_no_bias_non_power_of_2_width(self) -> None:
        torch.manual_seed(0)
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        x = (torch.randn(3, 8, 100) * 0.5)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False)
        assert out.shape == expected.shape == (4, 8, 100)
        assert torch.abs(out - expected).max() < _TOL

    def test_plain_conv_bias_relu_boundary_width_127(self) -> None:
        torch.manual_seed(3)
        layer = torch.nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
        x = (torch.randn(3, 4, 127) * 0.5)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=1, relu=True,
        )
        assert torch.abs(out - expected).max() < _TOL

    def test_depthwise_bias_relu_width_65(self) -> None:
        torch.manual_seed(1)
        c = 4
        layer = torch.nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=True)
        x = (torch.randn(c, 6, 65) * 0.5)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=c, relu=True,
        )
        assert torch.abs(out - expected).max() < _TOL

    def test_depthwise_no_bias_width_128(self) -> None:
        torch.manual_seed(2)
        c = 3
        layer = torch.nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        x = (torch.randn(c, 5, 128) * 0.5)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=c, relu=False)
        assert torch.abs(out - expected).max() < _TOL


class TestRunLayerStride2:

    def test_depthwise_stride2_multiple_of_128_width(self) -> None:
        torch.manual_seed(1)
        c = 4
        layer = torch.nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, groups=c, bias=False)
        x = (torch.randn(c, 8, 128) * 0.5)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=2, groups=c, relu=False)
        assert out.shape == expected.shape
        assert torch.abs(out - expected).max() < _TOL


class TestRunLayerPointwise:

    def test_plain_1x1_no_bias(self) -> None:
        torch.manual_seed(4)
        layer = torch.nn.Conv2d(16, 8, kernel_size=1, bias=False)
        x = (torch.randn(16, 16, 16) * 0.5)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False, padding=0)
        assert out.shape == expected.shape == (8, 16, 16)
        assert torch.abs(out - expected).max() < _TOL

    def test_1x1_bias_relu(self) -> None:
        torch.manual_seed(5)
        layer = torch.nn.Conv2d(16, 8, kernel_size=1, bias=True)
        x = (torch.randn(16, 16, 16) * 0.5)

        out = run_layer(layer, x, apply_relu=True)
        expected = _reference_conv(
            x, layer.weight.data, layer.bias.data, stride=1, groups=1, relu=True, padding=0,
        )
        assert torch.abs(out - expected).max() < _TOL

    def test_1x1_8x8_spatial_floor(self) -> None:
        torch.manual_seed(6)
        layer = torch.nn.Conv2d(160, 160, kernel_size=1, bias=False)
        x = (torch.randn(160, 8, 8) * 0.5)

        out = run_layer(layer, x)
        expected = _reference_conv(x, layer.weight.data, None, stride=1, groups=1, relu=False, padding=0)
        assert out.shape == expected.shape == (160, 8, 8)
        assert torch.abs(out - expected).max() < _TOL


class TestRegisterLayerAdapter:
    """The ``register_layer("Conv2d")`` adapter, via from_layer()."""

    def test_from_layer_routes_to_conv2d(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        op, params = from_layer(layer, (3, 8, 100))
        assert op == "conv2d"
        assert params["in_channels"] == 3
        assert params["out_channels"] == 4
        assert params["height"] == 8
        assert params["width"] == 100


class TestRefusals:

    def _depthwise_layer(self, **kw) -> torch.nn.Conv2d:
        c = kw.pop("channels", 4)
        return torch.nn.Conv2d(c, c, kernel_size=3, groups=c, **kw)

    def test_wrong_kernel_size(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=5, padding=2)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_dilation(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=2, dilation=2)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="dilation"):
            run_layer(layer, x)

    def test_non_same_padding(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=0)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="padding"):
            run_layer(layer, x)

    def test_unsupported_stride(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=3)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="stride"):
            run_layer(layer, x)

    def test_partial_groups(self) -> None:
        layer = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
        x = torch.zeros(4, 8, 8)
        with pytest.raises(UnsupportedLayer, match="groups"):
            run_layer(layer, x)

    def test_stride2_plain_conv_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=2)
        x = torch.zeros(3, 8, 128)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_bias_without_apply_relu_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="apply_relu"):
            run_layer(layer, x, apply_relu=False)

    def test_width_over_128_refused(self) -> None:
        # width=200 is not a multiple of 128 (>=384 requirement for wide384
        # either), so no k=3 app in this package covers it.
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        x = torch.zeros(3, 8, 200)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_non_square_stride_refused(self) -> None:
        layer = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=(1, 2), bias=False)
        x = torch.zeros(3, 8, 8)
        with pytest.raises(UnsupportedLayer, match="non-square"):
            run_layer(layer, x)

    def test_1x1_stride2_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, stride=2, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_1x1_groups_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, groups=8, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_1x1_in_channels_not_multiple_of_8_refused(self) -> None:
        layer = torch.nn.Conv2d(6, 8, kernel_size=1, bias=False)
        x = torch.zeros(6, 8, 8)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_1x1_out_channels_not_multiple_of_4_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 6, kernel_size=1, bias=False)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer):
            run_layer(layer, x)

    def test_1x1_bias_without_apply_relu_refused(self) -> None:
        layer = torch.nn.Conv2d(8, 8, kernel_size=1, bias=True)
        x = torch.zeros(8, 8, 8)
        with pytest.raises(UnsupportedLayer, match="apply_relu"):
            run_layer(layer, x, apply_relu=False)


class TestResolve:
    """Framework-free dispatch check (no torch layer needed, just the dataclass)."""

    def test_plain_conv_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=3, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, has_bias=False,
            height=16, width=16,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "conv_universal"

    def test_plain_conv_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=3, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, has_bias=True, apply_relu=True,
            height=16, width=16,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "conv_universal_bn_activation"

    def test_depthwise_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=4, has_bias=False,
            height=16, width=16,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "depthwise_conv_universal"

    def test_depthwise_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=4, has_bias=True, apply_relu=True,
            height=16, width=16,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "depthwise_conv_universal_bn_activation"

    def test_depthwise_stride2(self) -> None:
        desc = Conv2dDescription(
            in_channels=4, out_channels=4, kernel_size=3, stride=2,
            padding=1, dilation=1, groups=4, has_bias=False,
            height=8, width=128,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "depthwise_conv_stride2_128"

    def test_pointwise_no_bias(self) -> None:
        desc = Conv2dDescription(
            in_channels=8, out_channels=8, kernel_size=1, stride=1,
            padding=0, dilation=1, groups=1, has_bias=False,
            height=8, width=8,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "pointwise_conv_unified"

    def test_pointwise_bias_relu(self) -> None:
        desc = Conv2dDescription(
            in_channels=8, out_channels=8, kernel_size=1, stride=1,
            padding=0, dilation=1, groups=1, has_bias=True, apply_relu=True,
            height=8, width=8,
        )
        v = resolve(desc)
        assert v and v.kernel.name == "pointwise_conv_unified_bn_activation"
