"""The ``Conv2d`` layer adapter (``ipu_kernels.convolutions.layers``, E07-16):
``Registry.from_layer`` turns a ``torch.nn.Conv2d``-like layer into a
universal ``conv2d`` query, and refuses the configurations no kernel models.
A torch-free stand-in for the layer is enough: adapters match by class name.
"""

from __future__ import annotations

import pytest

from ipu_harness.registry import UnsupportedLayer, default_registry
from ipu_kernels.convolutions.layers import from_torch_conv2d


class Conv2d:
    """Stand-in for ``torch.nn.Conv2d``: the same attributes, no torch."""

    DEFAULTS = dict(stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")

    def __init__(self, in_channels, out_channels, kernel_size, **options):
        options = self.DEFAULTS | options
        pair = lambda v: v if isinstance(v, (tuple, str)) else (v, v)  # noqa: E731
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = pair(kernel_size)
        self.stride, self.padding = pair(options["stride"]), pair(options["padding"])
        self.dilation = pair(options["dilation"])
        self.groups, self.padding_mode = options["groups"], options["padding_mode"]
        self.bias = object() if options["bias"] else None


@pytest.fixture(scope="module")
def registry():
    return default_registry()


def test_conv2d_adapter_is_registered(registry):
    assert "Conv2d" in registry.adapters


def test_the_adapter_makes_a_universal_query():
    geometry = from_torch_conv2d(Conv2d(3, 4, kernel_size=3, padding=1, bias=False), (3, 8, 100))
    assert geometry.params() == dict(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        has_bias=False,
        apply_relu=False,
        height=8,
        width=100,
    )


def test_relu_is_reached_only_through_an_explicit_geometry(registry):
    # The first layer applies ReLU; the layer alone never implies it...
    layer = Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=True)
    assert not registry.from_layer(layer, (3, 256, 256))
    # ...and asking for it explicitly routes to the kernel.
    geometry = from_torch_conv2d(layer, (3, 256, 256), apply_relu=True)
    assert registry.resolve("conv2d", **geometry.params()).kernel_name == "conv_first_layer"


@pytest.mark.parametrize(
    "layer,shape,reason",
    [
        (
            Conv2d(3, 4, kernel_size=3, padding=1, stride=(1, 2), bias=False),
            (3, 8, 8),
            "non-square",
        ),
        (Conv2d(3, 4, kernel_size=3, padding="same", bias=False), (3, 8, 8), "padding='same'"),
        (Conv2d(3, 4, kernel_size=3, padding=1, padding_mode="reflect"), (3, 8, 8), "padding_mode"),
        (Conv2d(3, 4, kernel_size=3, padding=1), (1, 3, 8, 8), "no batch dimension"),
        (Conv2d(3, 4, kernel_size=3, padding=1), (4, 8, 8), "in_channels=3"),
    ],
)
def test_unmodelled_layer_configurations_are_refused(registry, layer, shape, reason):
    with pytest.raises(UnsupportedLayer, match=reason):
        registry.from_layer(layer, shape)


def test_from_layer_routes_to_conv_universal(registry):
    layer = Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    op, params = registry.adapters["Conv2d"](layer, (3, 8, 100))
    assert op == "conv2d"
    assert (params["in_channels"], params["out_channels"]) == (3, 4)
    assert (params["height"], params["width"]) == (8, 100)
    assert registry.from_layer(layer, (3, 8, 100)).kernel_name == "conv_universal"


def test_a_biased_layer_routes_to_the_bias_kernel_and_relu_is_explicit(registry):
    # A biased layer alone is a bias without ReLU: conv_universal_bias (E07-20;
    # refused before it)...
    layer = Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
    verdict = registry.from_layer(layer, (3, 8, 8))
    assert verdict.kernel_name == "conv_universal_bias", verdict.reason
    # ...and the bias + ReLU twin is reached through an explicit geometry.
    geometry = from_torch_conv2d(layer, (3, 8, 8), apply_relu=True)
    verdict = registry.resolve("conv2d", **geometry.params())
    assert verdict.kernel_name == "conv_universal_bn_activation"


def test_the_depthwise_relu_twin_is_reached_through_an_explicit_geometry(registry):
    layer = Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=True)
    # The layer alone never implies ReLU: the depthwise bias kernel (E07-20)...
    verdict = registry.from_layer(layer, (4, 6, 65))
    assert verdict.kernel_name == "depthwise_conv_universal_bias", verdict.reason
    # ...and asking for it explicitly routes to the depthwise bias + ReLU twin.
    geometry = from_torch_conv2d(layer, (4, 6, 65), apply_relu=True)
    verdict = registry.resolve("conv2d", **geometry.params())
    assert verdict.kernel_name == "depthwise_conv_universal_bn_activation"


@pytest.mark.parametrize(
    "layer,shape,kernel",
    [
        (Conv2d(16, 16, kernel_size=1, bias=False), (16, 8, 8), "pointwise_conv_unified"),
        (
            Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False),
            (4, 8, 8),
            "depthwise_conv_universal",
        ),
        # A bias without ReLU (E07-20; refused before it).
        (Conv2d(8, 8, kernel_size=1, bias=True), (8, 8, 8), "pointwise_conv_unified_bias"),
    ],
)
def test_from_layer_routes_to_the_pointwise_and_depthwise_kernels(registry, layer, shape, kernel):
    assert registry.from_layer(layer, shape).kernel_name == kernel


@pytest.mark.parametrize(
    "layer,shape,reason",
    [
        (
            Conv2d(16, 16, kernel_size=1, bias=False),
            (16, 8, 129),
            "width = 129 is above its max 128",
        ),
        (Conv2d(8, 8, kernel_size=1, stride=2, bias=False), (8, 8, 8), "stride"),
        (Conv2d(8, 8, kernel_size=1, groups=8, bias=False), (8, 8, 8), "groups"),
        (Conv2d(6, 8, kernel_size=1, bias=False), (6, 8, 8), "in_channels = 6 is below its min 8"),
        (
            Conv2d(8, 6, kernel_size=1, bias=False),
            (8, 8, 8),
            "out_channels = 6 is not a multiple of 4",
        ),
    ],
)
def test_pointwise_layer_refusals(registry, layer, shape, reason):
    verdict = registry.from_layer(layer, shape)
    assert not verdict
    assert reason in verdict.reason, verdict.reason
