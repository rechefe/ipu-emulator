"""Routing of the convolutions (E07-16): the old router tests for the
kernels migrated so far, on the kernel tree's registry and ``//kernels:query``.

Two ``conv2d`` vocabularies share the op. The tiled kernels (``conv1x1``,
``conv3x3_relu``, ``conv3x3_relu_cin1``) take ``in_channels``, ``height``,
``width``, ``out_channels``, ``kernel_size``, ``stride``, ``padding`` and
``activation``, every one required (the old harness named the image ``shape=(c,
h, w)``, a tuple no ``kernel.toml`` parameter can be). The universal kernels
(``conv_first_layer``; ``conv_universal`` and ``conv_universal_bn_activation``,
E07-17; the pointwise and depthwise twins, E07-18; ``conv_universal_wide384``
and the ``depthwise_conv_stride2_*`` kernels, E07-19; the bias-without-ReLU
entries, E07-20) take the eleven fields of a
``torch.nn.Conv2d`` query, ``dilation``, ``groups``, ``has_bias`` and ``apply_relu`` among them.
A query in one vocabulary names a parameter the other's kernels lack, so it
reaches only its own kernels.
"""

from __future__ import annotations

import json

import pytest

from ipu_harness import query
from ipu_harness.params import plan_case
from ipu_harness.registry import default_registry
from ipu_isa.formats import ElementFormat

TILED = {"conv1x1", "conv3x3_relu", "conv3x3_relu_cin1"}

FIRST_LAYER = dict(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1,
    dilation=1,
    groups=1,
    has_bias=True,
    apply_relu=True,
    height=256,
    width=256,
)


def tiled(k: int, **overrides):
    """A tiled-vocabulary query for a ``k`` x ``k`` kernel (padding k // 2)."""
    params = dict(
        in_channels=3,
        height=3,
        width=8,
        out_channels=2,
        kernel_size=k,
        stride=1,
        padding=k // 2,
        activation="none" if k == 1 else "relu",
    )
    return params | overrides


@pytest.fixture(scope="module")
def registry():
    loaded = default_registry()
    assert not loaded.invalid, loaded.invalid
    return loaded


def test_both_conv2d_vocabularies_resolve_to_their_own_kernels(registry):
    universal = registry.resolve("conv2d", **FIRST_LAYER)
    assert universal.kernel_name == "conv_first_layer", universal.reason
    legacy = registry.resolve("conv2d", **tiled(3))
    assert legacy.kernel_name == "conv3x3_relu", legacy.reason
    assert not TILED & set(universal.alternatives)
    assert set(legacy.alternatives) <= TILED


def test_a_universal_query_names_parameters_no_tiled_kernel_takes(registry):
    # The tiled kernels' shape, asked in the universal vocabulary: every tiled
    # kernel refuses it by name (a universal kernel takes it, E07-17).
    params = tiled(3) | dict(dilation=1, groups=1, has_bias=True, apply_relu=True)
    del params["activation"]
    verdict = registry.resolve("conv2d", **params)
    assert verdict.kernel_name == "conv_universal_bn_activation", verdict.reason
    assert not TILED & set(verdict.alternatives)
    for name in TILED:
        assert registry.kernel(name).params.keys().isdisjoint({"dilation", "groups", "has_bias"})


@pytest.mark.parametrize(
    "params,kernel",
    [
        (tiled(1), "conv1x1"),
        (tiled(1, in_channels=129, width=133), "conv1x1"),
        (tiled(3), "conv3x3_relu"),
        (tiled(3, in_channels=15), "conv3x3_relu"),
        (tiled(3, in_channels=1), "conv3x3_relu_cin1"),
        (FIRST_LAYER, "conv_first_layer"),
    ],
)
def test_resolve(registry, params, kernel):
    verdict = registry.resolve("conv2d", **params)
    assert verdict.kernel_name == kernel, verdict.reason
    assert verdict.label == f"//kernels/convolutions/{kernel}"


def test_one_input_channel_prefers_the_cin1_kernel(registry):
    verdict = registry.resolve("conv2d", **tiled(3, in_channels=1))
    assert verdict.kernel_name == "conv3x3_relu_cin1"
    assert verdict.alternatives == ("conv3x3_relu",)


@pytest.mark.parametrize("missing", ["kernel_size", "stride", "padding", "activation"])
def test_a_tiled_query_must_name_every_parameter(registry, missing):
    # As the old harness's requires: a parameter left out is refused, not defaulted.
    params = tiled(3)
    del params[missing]
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert missing in verdict.reason


@pytest.mark.parametrize(
    "params,reason",
    [
        (tiled(3, activation="none"), "activation"),  # a 3x3 tiled kernel always applies ReLU
        (tiled(1, activation="relu"), "activation"),  # ... and the 1x1 one never does
        (tiled(3, stride=2), "stride"),
        (tiled(3, padding=0), "padding"),
        (tiled(5, padding=2), "kernel_size"),
        (tiled(3, in_channels=0), "in_channels"),
    ],
)
def test_tiled_refusals(registry, params, reason):
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert reason in verdict.reason


@pytest.mark.parametrize(
    "change",
    [dict(height=128), dict(out_channels=8), dict(stride=1), dict(apply_relu=False)],
)
def test_the_first_layer_kernel_takes_only_its_fixed_shape(registry, change):
    assert not registry.resolve("conv2d", **(FIRST_LAYER | change))


def test_the_cin1_kernel_takes_one_input_channel_only(registry):
    cin1 = registry.kernel("conv3x3_relu_cin1")
    assert cin1.params["in_channels"].choices == (1,)


@pytest.mark.parametrize(
    "params,kernel",
    [(tiled(1), "conv1x1"), (tiled(3), "conv3x3_relu"), (FIRST_LAYER, "conv_first_layer")],
)
def test_the_query_tool_gives_the_same_answer(registry, params, kernel, capsys):
    argv = [
        "conv2d",
        *(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in params.items()),
    ]
    assert query.main([*argv, "--json"], registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["app"], answer["use"]) == (kernel, f"//kernels/convolutions/{kernel}")


# -- the universal 3x3 kernels (E07-17): conv_universal and its bias + ReLU twin


def universal(**overrides):
    """A universal-vocabulary query for the 3x3 stride-1 same-padding kernels."""
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


@pytest.mark.parametrize(
    "has_bias,apply_relu,kernel",
    [
        (False, False, "conv_universal"),
        (True, True, "conv_universal_bn_activation"),
        (False, True, "conv_universal_bn_activation"),  # a zero bias
        (True, False, "conv_universal_bias"),  # refused before E07-20
    ],
)
def test_bias_and_relu_choose_the_universal_kernel(registry, has_bias, apply_relu, kernel):
    verdict = registry.resolve("conv2d", **universal(has_bias=has_bias, apply_relu=apply_relu))
    assert verdict.kernel_name == kernel, verdict.reason
    if kernel is not None:
        assert verdict.label == f"//kernels/convolutions:{kernel}"


@pytest.mark.parametrize("missing", list(universal()))
def test_a_universal_query_must_name_every_parameter(registry, missing):
    # As the old harness's requires: a parameter left out is refused, not defaulted.
    params = universal()
    del params[missing]
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert missing in verdict.reason


@pytest.mark.parametrize(
    "params,reason",
    [
        (universal(kernel_size=5, padding=2), "kernel_size"),
        (universal(dilation=2, padding=2), "dilation"),
        (universal(padding=0), "padding"),
        (universal(stride=3), "stride"),
        (universal(in_channels=4, groups=2), "groups"),
        (universal(width=200), "width"),
        (universal(in_channels=0), "in_channels"),
    ],
)
def test_universal_refusals(registry, params, reason):
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert reason in verdict.reason


@pytest.mark.parametrize("width", [1, 16, 17, 100, 127, 128])
def test_widths_up_to_128_route_to_conv_universal(registry, width):
    assert registry.resolve("conv2d", **universal(width=width)).kernel_name == "conv_universal"


def test_width_129_is_refused(registry):
    # conv_universal_wide384 takes only multiples of 128 from 384 up.
    verdict = registry.resolve("conv2d", **universal(width=129))
    assert not verdict
    assert "width" in verdict.reason


@pytest.mark.parametrize(
    "params,kernel",
    [
        (universal(), "conv_universal"),
        (universal(has_bias=True, apply_relu=True), "conv_universal_bn_activation"),
        (universal(has_bias=True), "conv_universal_bias"),
    ],
)
def test_the_query_tool_names_the_entry(registry, params, kernel, capsys):
    argv = [
        "conv2d",
        *(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in params.items()),
    ]
    assert query.main([*argv, "--json"], registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["app"], answer["use"]) == (kernel, f"//kernels/convolutions:{kernel}")


# -- the pointwise and depthwise twins (E07-18)


def pointwise(**overrides):
    """A universal-vocabulary query for the 1x1 kernels."""
    return (
        universal(in_channels=8, out_channels=8, kernel_size=1, padding=0, height=8, width=8)
        | overrides
    )


def depthwise(**overrides):
    """A universal-vocabulary query for the depthwise 3x3 stride-1 kernels."""
    return universal(in_channels=4, out_channels=4, groups=4) | overrides


@pytest.mark.parametrize(
    "has_bias,apply_relu,kernel",
    [
        (False, False, "pointwise_conv_unified"),
        (True, True, "pointwise_conv_unified_bn_activation"),
        (False, True, "pointwise_conv_unified_bn_activation"),  # a zero bias
        (True, False, "pointwise_conv_unified_bias"),  # refused before E07-20
    ],
)
def test_bias_and_relu_choose_the_pointwise_kernel(registry, has_bias, apply_relu, kernel):
    verdict = registry.resolve("conv2d", **pointwise(has_bias=has_bias, apply_relu=apply_relu))
    assert verdict.kernel_name == kernel, verdict.reason
    if kernel is not None:
        assert verdict.label == f"//kernels/convolutions:{kernel}"


@pytest.mark.parametrize(
    "has_bias,apply_relu,kernel",
    [
        (False, False, "depthwise_conv_universal"),
        (True, True, "depthwise_conv_universal_bn_activation"),
        (False, True, "depthwise_conv_universal_relu"),  # a zero bias; refused before E07-20
        (True, False, "depthwise_conv_universal_bias"),  # refused before E07-20
    ],
)
def test_bias_and_relu_choose_the_depthwise_kernel(registry, has_bias, apply_relu, kernel):
    verdict = registry.resolve("conv2d", **depthwise(has_bias=has_bias, apply_relu=apply_relu))
    assert verdict.kernel_name == kernel, verdict.reason
    if kernel is not None:
        assert verdict.label == f"//kernels/convolutions:{kernel}"


@pytest.mark.parametrize("make", [universal, pointwise, depthwise])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("apply_relu", [False, True])
def test_every_bias_and_relu_combination_routes(registry, make, has_bias, apply_relu):
    # E07-20: each template's geometry answers all four combinations, each by
    # one entry (has_bias and apply_relu split the domains, so no tie).
    verdict = registry.resolve("conv2d", **make(has_bias=has_bias, apply_relu=apply_relu))
    assert verdict, verdict.reason
    assert verdict.label == f"//kernels/convolutions:{verdict.kernel_name}"


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("apply_relu", [False, True])
def test_one_channel_depthwise_is_a_plain_3x3_first_at_every_combination(
    registry, has_bias, apply_relu
):
    # The C = 1 query is also depthwise: conv_universal's entries cost less.
    verdict = registry.resolve(
        "conv2d",
        **universal(in_channels=1, out_channels=1, has_bias=has_bias, apply_relu=apply_relu),
    )
    assert verdict.kernel_name.startswith("conv_universal"), verdict.reason
    assert any(name.startswith("depthwise_conv_universal") for name in verdict.alternatives)


def test_one_channel_depthwise_is_a_plain_3x3_first(registry):
    # in_channels = out_channels = groups = 1 is both: the plain kernel costs less.
    verdict = registry.resolve("conv2d", **universal(in_channels=1, out_channels=1))
    assert verdict.kernel_name == "conv_universal", verdict.reason
    assert "depthwise_conv_universal" in verdict.alternatives


@pytest.mark.parametrize("make", [pointwise, depthwise])
@pytest.mark.parametrize("missing", list(universal()))
def test_a_pointwise_or_depthwise_query_must_name_every_parameter(registry, make, missing):
    params = make()
    del params[missing]
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert missing in verdict.reason


@pytest.mark.parametrize(
    "params,reason",
    [
        (pointwise(width=129), "width = 129 is above its max 128"),
        (pointwise(stride=2), "stride"),
        (pointwise(groups=8), "groups"),
        (pointwise(in_channels=6), "in_channels = 6 is below its min 8"),
        (pointwise(in_channels=12), "in_channels = 12 is not a multiple of 8"),
        (pointwise(out_channels=6), "out_channels = 6 is not a multiple of 4"),
        (depthwise(out_channels=8), "out_channels"),
        (depthwise(groups=2), "groups"),
        (depthwise(width=129), "width = 129 is above its max 128"),
    ],
)
def test_pointwise_and_depthwise_refusals(registry, params, reason):
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert reason in verdict.reason, verdict.reason


@pytest.mark.parametrize(
    "params,kernel",
    [
        (pointwise(), "pointwise_conv_unified"),
        (depthwise(has_bias=True, apply_relu=True), "depthwise_conv_universal_bn_activation"),
        (pointwise(has_bias=True), "pointwise_conv_unified_bias"),
        (depthwise(has_bias=True), "depthwise_conv_universal_bias"),
    ],
)
def test_the_query_tool_names_the_pointwise_and_depthwise_entries(registry, params, kernel, capsys):
    argv = [
        "conv2d",
        *(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in params.items()),
    ]
    assert query.main([*argv, "--json"], registry) == 0
    answer = json.loads(capsys.readouterr().out)
    assert (answer["app"], answer["use"]) == (kernel, f"//kernels/convolutions:{kernel}")


# -- conv_universal_wide384 and the stride-2 depthwise kernels (E07-19)


def stride2(**overrides):
    """A universal-vocabulary query for the 3x3 stride-2 depthwise kernels."""
    return universal(in_channels=4, out_channels=4, groups=4, stride=2) | overrides


@pytest.mark.parametrize(
    "params,kernel",
    [
        (stride2(height=8, width=128), "depthwise_conv_stride2_128"),
        (stride2(height=16, width=32), "depthwise_conv_stride2_narrow"),
        (stride2(height=16, width=16), "depthwise_conv_stride2_16"),
        (universal(width=384, out_channels=4), "conv_universal_wide384"),
    ],
)
def test_resolve_wide_and_stride2(registry, params, kernel):
    verdict = registry.resolve("conv2d", **params)
    assert verdict.kernel_name == kernel, verdict.reason
    assert verdict.label == f"//kernels/convolutions/{kernel}"


def _width_runs(registry, widths, **fixed):
    """``(first, last, kernel)`` for each run of widths one kernel (or none) takes."""
    runs: list[list] = []
    for width in widths:
        verdict = registry.resolve("conv2d", **(fixed | dict(width=width)))
        name = verdict.kernel_name if verdict else None
        if runs and runs[-1][2] == name and runs[-1][1] == width - 1:
            runs[-1][1] = width
        else:
            runs.append([width, width, name])
    return [tuple(run) for run in runs]


def test_width_routing_boundaries(registry):
    # The old router's boundary table: conv_universal up to 128, then
    # conv_universal_wide384 at each multiple of 128 from 384.
    assert _width_runs(registry, range(1, 641), **universal(out_channels=4)) == [
        (1, 128, "conv_universal"),
        (129, 383, None),
        (384, 384, "conv_universal_wide384"),
        (385, 511, None),
        (512, 512, "conv_universal_wide384"),
        (513, 639, None),
        (640, 640, "conv_universal_wide384"),
    ]


@pytest.mark.parametrize(
    "params,reason",
    [
        # depthwise_conv_stride2_narrow: the output groups of 4 chunks must be whole ...
        (
            stride2(in_channels=1, out_channels=1, groups=1, height=5, width=64),
            "is 5/8, not a whole number",
        ),
        # ... (width 16: rows_per_chunk 8 needs height % 32 == 0), and
        # depthwise_conv_stride2_16 packs channels in pairs.
        (
            stride2(in_channels=1, out_channels=1, groups=1, height=16, width=16),
            "is 1/2, not a whole number",
        ),
        (
            stride2(in_channels=3, out_channels=3, groups=3, height=16, width=16),
            "in_channels = 3 is not a multiple of 2",
        ),
        (stride2(height=8, width=48), "width = 48 is not one of [16, 32, 64]"),
        # depthwise_conv_stride2_128: output rows pair up
        (
            stride2(in_channels=1, out_channels=1, groups=1, height=5, width=128),
            "height = 5 is not a multiple of 4",
        ),
        (stride2(has_bias=True, apply_relu=True, height=8, width=128), "has_bias"),
        # conv_universal_wide384: out_channels in pairs, widths multiples of 128 from 384
        (
            universal(in_channels=1, out_channels=3, height=8, width=384),
            "out_channels = 3 is not a multiple of 2",
        ),
        (universal(width=385), "width = 385 is not a multiple of 128"),
        (universal(width=200), "width"),
    ],
)
def test_wide_and_stride2_refusals(registry, params, reason):
    verdict = registry.resolve("conv2d", **params)
    assert not verdict
    assert reason in verdict.reason, verdict.reason


def test_the_stride2_output_shape_is_derived(registry):
    kernel = registry.kernel("depthwise_conv_stride2_128")
    plan = plan_case(kernel, stride2(height=8, width=128), ElementFormat.FP32)
    assert plan.shapes["out"] == (4, 4, 64)
