"""Convolution family tests, beyond each kernel's own cases and test.py.

* **Two conv2d vocabularies** -- the universal kernels (``in_channels`` /
  ``height`` / ``width`` / ``groups`` / ``has_bias`` / ``apply_relu`` ...)
  and the memory-layout ``conv1x1`` / ``conv3x3_relu*`` kernels (``shape`` /
  ``activation`` ...) both answer ``resolve("conv2d", ...)``; ``requires`` keeps each vocabulary routed to its own kernels.
* **Routing** -- framework-free dispatch checks and the width ceiling, as a
  generated boundary table.
* **The ``Conv2d`` layer adapter** -- refusal cases, driven through
  :func:`~ipu_apps.kernel_registry.lookup_layer` with a torch-free stand-in
  for ``torch.nn.Conv2d`` (adapters match by class name), including
  ``pointwise_conv_unified``'s width bound (a width > 128 pointwise query must
  be refused, since ``pointwise_pad_shape`` has no divisor of 128 above 128).
"""
from __future__ import annotations

import pytest

from ipu_apps.kernel_registry import (
    UnsupportedLayer, adapters, boundaries, from_layer, lookup_layer, resolve,
)
from ipu_apps.kernels.convolutions.app import (
    Conv2dDescription, from_torch_conv2d, pointwise_pad_shape,
)


class Conv2d:
    """Stand-in for ``torch.nn.Conv2d``: the same attributes, no torch."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros"):
        pair = lambda v: v if isinstance(v, (tuple, str)) else (v, v)  # noqa: E731
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = pair(kernel_size), pair(stride)
        self.padding, self.dilation = pair(padding), pair(dilation)
        self.groups, self.padding_mode = groups, padding_mode
        self.bias = object() if bias else None


def query(**overrides):
    params = dict(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1,
                  dilation=1, groups=1, has_bias=False, apply_relu=False,
                  height=16, width=16)
    return params | overrides


# -- two vocabularies -----------------------------------------------------------


def test_both_conv2d_vocabularies_resolve_to_their_own_kernels():
    universal = resolve("conv2d", **query(in_channels=3, out_channels=2, height=3, width=8,
                                          has_bias=True, apply_relu=True))
    assert universal.app_name == "conv_universal_bn_activation"
    legacy = resolve("conv2d", shape=(3, 3, 8), out_channels=2, kernel_size=3,
                     stride=1, padding=1, activation="relu")
    assert legacy.app_name == "conv3x3_relu"
    legacy_names = {"conv1x1", "conv3x3_relu", "conv3x3_relu_cin1"}
    assert not legacy_names & set(universal.alternatives)
    assert not {a for a in legacy.alternatives if a not in legacy_names}


# -- routing -------------------------------------------------------------------


@pytest.mark.parametrize("overrides,kernel", [
    (dict(), "conv_universal"),
    (dict(has_bias=True, apply_relu=True), "conv_universal_bn_activation"),
    (dict(in_channels=4, out_channels=4, groups=4), "depthwise_conv_universal"),
    (dict(in_channels=4, out_channels=4, groups=4, has_bias=True, apply_relu=True),
     "depthwise_conv_universal_bn_activation"),
    (dict(in_channels=4, out_channels=4, groups=4, stride=2, height=8, width=128),
     "depthwise_conv_stride2_128"),
    (dict(in_channels=4, out_channels=4, groups=4, stride=2, height=16, width=32),
     "depthwise_conv_stride2_narrow"),
    (dict(in_channels=4, out_channels=4, groups=4, stride=2, height=16, width=16),
     "depthwise_conv_stride2_16"),
    (dict(in_channels=8, out_channels=8, kernel_size=1, padding=0, height=8, width=8),
     "pointwise_conv_unified"),
    (dict(in_channels=8, out_channels=8, kernel_size=1, padding=0, height=8, width=8,
          has_bias=True, apply_relu=True), "pointwise_conv_unified_bn_activation"),
    (dict(width=384, out_channels=4), "conv_universal_wide384"),
    (dict(in_channels=3, out_channels=16, stride=2, height=256, width=256,
          has_bias=True, apply_relu=True), "conv_first_layer"),
])
def test_resolve(overrides, kernel):
    desc = Conv2dDescription(**query(**overrides))
    verdict = resolve("conv2d", **desc.params())
    assert verdict and verdict.app_name == kernel, verdict.reason


def test_width_routing_boundaries():
    fixed = query(out_channels=4)
    del fixed["width"]
    runs = boundaries("conv2d", "width", range(1, 641), **fixed)
    assert [(b.start, b.end, b.kernel) for b in runs] == [
        (1, 128, "conv_universal"),
        (129, 383, None),
        (384, 384, "conv_universal_wide384"),
        (385, 511, None),
        (512, 512, "conv_universal_wide384"),
        (513, 639, None),
        (640, 640, "conv_universal_wide384"),
    ]


def test_output_shape_is_derived_for_stride2():
    verdict = resolve("conv2d", **query(in_channels=4, out_channels=4, groups=4, stride=2,
                                        height=8, width=128))
    assert verdict.shapes.get("output") == (4, 4, 64)
    assert "output" in verdict.shapes.derived_roles


# -- the Conv2d adapter refusals ------------------------------------------------


def test_conv2d_adapter_is_registered():
    assert "Conv2d" in adapters()


def test_from_layer_routes_to_conv2d():
    op, params = from_layer(Conv2d(3, 4, kernel_size=3, padding=1, bias=False), (3, 8, 100))
    assert op == "conv2d"
    assert (params["in_channels"], params["out_channels"]) == (3, 4)
    assert (params["height"], params["width"]) == (8, 100)
    assert lookup_layer(Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
                        (3, 8, 100)).app_name == "conv_universal"


def test_relu_twins_are_reached_through_an_explicit_description():
    layer = Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=True)
    # The layer alone never implies ReLU, so the bias-capable twins refuse it...
    assert not lookup_layer(layer, (4, 6, 65))
    # ...and asking for it explicitly routes to the depthwise BN twin.
    desc = from_torch_conv2d(layer, (4, 6, 65), apply_relu=True)
    assert resolve("conv2d", **desc.params()).app_name == "depthwise_conv_universal_bn_activation"


@pytest.mark.parametrize("layer,shape,reason", [
    (Conv2d(3, 4, kernel_size=5, padding=2), (3, 8, 8), "kernel_size"),
    (Conv2d(3, 4, kernel_size=3, padding=2, dilation=2), (3, 8, 8), "dilation"),
    (Conv2d(3, 4, kernel_size=3, padding=0), (3, 8, 8), "padding"),
    (Conv2d(3, 4, kernel_size=3, padding=1, stride=3), (3, 8, 8), "stride"),
    (Conv2d(4, 4, kernel_size=3, padding=1, groups=2), (4, 8, 8), "groups"),
    (Conv2d(3, 4, kernel_size=3, padding=1, stride=2), (3, 8, 128), "stride"),
    (Conv2d(3, 4, kernel_size=3, padding=1, bias=True), (3, 8, 8), "apply_relu"),
    # width 200: over 128 and not a multiple of 128 >= 384 for wide384 either.
    (Conv2d(3, 4, kernel_size=3, padding=1, bias=False), (3, 8, 200), "width"),
    # Refused by the pointwise width bound before pointwise_pad_shape is reached.
    (Conv2d(16, 16, kernel_size=1, bias=False), (16, 8, 129), "width (129) exceeds 128"),
    (Conv2d(8, 8, kernel_size=1, stride=2, bias=False), (8, 8, 8), "stride"),
    (Conv2d(8, 8, kernel_size=1, groups=8, bias=False), (8, 8, 8), "groups"),
    (Conv2d(6, 8, kernel_size=1, bias=False), (6, 8, 8), "multiple of 8"),
    (Conv2d(8, 6, kernel_size=1, bias=False), (8, 8, 8), "multiple of 4"),
    (Conv2d(8, 8, kernel_size=1, bias=True), (8, 8, 8), "apply_relu"),
])
def test_refusals(layer, shape, reason):
    verdict = lookup_layer(layer, shape)
    assert not verdict
    assert reason in verdict.reason


@pytest.mark.parametrize("layer,shape,reason", [
    (Conv2d(3, 4, kernel_size=3, padding=1, stride=(1, 2), bias=False), (3, 8, 8), "non-square"),
    (Conv2d(3, 4, kernel_size=3, padding="same", bias=False), (3, 8, 8), "padding='same'"),
    (Conv2d(3, 4, kernel_size=3, padding=1, padding_mode="reflect"), (3, 8, 8), "padding_mode"),
    (Conv2d(3, 4, kernel_size=3, padding=1), (1, 3, 8, 8), "no batch dimension"),
    (Conv2d(3, 4, kernel_size=3, padding=1), (4, 8, 8), "in_channels=3"),
])
def test_unmodelled_layer_configurations_are_refused(layer, shape, reason):
    with pytest.raises(UnsupportedLayer, match=reason):
        from_layer(layer, shape)


def test_pointwise_pad_shape_refuses_width_over_128():
    with pytest.raises(ValueError, match="no divisor of 128"):
        pointwise_pad_shape(8, 129)
