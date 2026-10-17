"""Convolution family tests, beyond each kernel's own cases and test.py.

* The tiled kernels (``conv1x1``, ``conv3x3_relu*``), ``conv_first_layer``
  (E07-16), ``conv_universal`` / ``conv_universal_bn_activation`` (E07-17)
  and the pointwise and depthwise twins (E07-18) moved to
  ``kernels/convolutions``; their routing and the two ``conv2d`` vocabularies
  are tested there (``test_routing.py``, ``test_layers.py``). The old
  ``conv_universal`` folder stays frozen for the Cycle Inspector (E07-26), so
  the width table below still names it, and the old
  ``depthwise_conv_universal`` folder as stage 1 of the stride-2 kernels
  (E07-19).
* **Routing** -- framework-free dispatch checks and the width ceiling, as a
  generated boundary table.
* **The ``Conv2d`` layer adapter** -- refusal cases, driven through
  :func:`~ipu_apps.kernel_registry.lookup_layer` with a torch-free stand-in
  for ``torch.nn.Conv2d`` (adapters match by class name).
"""

from __future__ import annotations

import pytest

from ipu_apps.kernel_registry import (
    UnsupportedLayer,
    adapters,
    boundaries,
    from_layer,
    lookup_layer,
    resolve,
)
from ipu_apps.kernels.convolutions.app import Conv2dDescription


class Conv2d:
    """Stand-in for ``torch.nn.Conv2d``: the same attributes, no torch."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        pair = lambda v: v if isinstance(v, (tuple, str)) else (v, v)  # noqa: E731
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = pair(kernel_size), pair(stride)
        self.padding, self.dilation = pair(padding), pair(dilation)
        self.groups, self.padding_mode = groups, padding_mode
        self.bias = object() if bias else None


def query(**overrides):
    params = dict(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        has_bias=False,
        apply_relu=False,
        height=16,
        width=16,
    )
    return params | overrides


# -- routing -------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,kernel",
    [
        (dict(in_channels=4, out_channels=4, groups=4), "depthwise_conv_universal"),
        (
            dict(in_channels=4, out_channels=4, groups=4, stride=2, height=8, width=128),
            "depthwise_conv_stride2_128",
        ),
        (
            dict(in_channels=4, out_channels=4, groups=4, stride=2, height=16, width=32),
            "depthwise_conv_stride2_narrow",
        ),
        (
            dict(in_channels=4, out_channels=4, groups=4, stride=2, height=16, width=16),
            "depthwise_conv_stride2_16",
        ),
        (dict(width=384, out_channels=4), "conv_universal_wide384"),
    ],
)
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
    verdict = resolve(
        "conv2d", **query(in_channels=4, out_channels=4, groups=4, stride=2, height=8, width=128)
    )
    assert verdict.shapes.get("output") == (4, 4, 64)
    assert "output" in verdict.shapes.derived_roles


# -- the Conv2d adapter refusals ------------------------------------------------


def test_conv2d_adapter_is_registered():
    assert "Conv2d" in adapters()


@pytest.mark.parametrize(
    "layer,shape,reason",
    [
        (Conv2d(3, 4, kernel_size=5, padding=2), (3, 8, 8), "kernel_size"),
        (Conv2d(3, 4, kernel_size=3, padding=2, dilation=2), (3, 8, 8), "dilation"),
        (Conv2d(3, 4, kernel_size=3, padding=0), (3, 8, 8), "padding"),
        (Conv2d(3, 4, kernel_size=3, padding=1, stride=3), (3, 8, 8), "stride"),
        (Conv2d(4, 4, kernel_size=3, padding=1, groups=2), (4, 8, 8), "groups"),
        (Conv2d(3, 4, kernel_size=3, padding=1, stride=2), (3, 8, 128), "stride"),
        # width 200: over 128 and not a multiple of 128 >= 384 for wide384 either.
        (Conv2d(3, 4, kernel_size=3, padding=1, bias=False), (3, 8, 200), "width"),
    ],
)
def test_refusals(layer, shape, reason):
    verdict = lookup_layer(layer, shape)
    assert not verdict
    assert reason in verdict.reason


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
def test_unmodelled_layer_configurations_are_refused(layer, shape, reason):
    with pytest.raises(UnsupportedLayer, match=reason):
        from_layer(layer, shape)
