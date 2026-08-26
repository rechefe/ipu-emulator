"""Conv2d-flavoured front end over the generic kernel registry.

This module is a **thin translation layer**, kept so callers get a
`catalog()`/CLI-friendly surface without having to build a
:class:`~ipu_apps.convolutions_universal.layers.Conv2dDescription` by hand.
It holds no routing rules of its own -- every answer comes from
:mod:`ipu_apps.kernel_registry`, which asks the kernels themselves.

    >>> from ipu_apps.convolutions_universal import lookup
    >>> lookup(in_channels=32, out_channels=32, kernel_size=3, stride=1,
    ...        padding=1, groups=32, height=64, width=64).app_name
    'depthwise_conv_universal'

Prefer the generic interface for new code -- it takes a framework layer, or an
op plus shapes, and works for every operation rather than just conv2d::

    from ipu_apps.kernel_registry import lookup_layer, resolve

    lookup_layer(nn.Conv2d(32, 32, 3, groups=32), input_shape=(32, 64, 64))
    resolve("conv2d", in_channels=32, out_channels=32, kernel_size=3, ...)

:data:`GAPS` lists shapes no kernel covers, worded as concretely as possible.
Unlike softmax (every 1-D shape is routable), convolution coverage genuinely
has holes -- e.g. arbitrary `kernel_size` values other than 1 and 3 have no
kernel at all -- so this list is not empty.
"""

from __future__ import annotations

from ipu_apps.kernel_registry import Verdict, boundaries, kernels, resolve
from ipu_apps.convolutions_universal.layers import Conv2dDescription, from_torch_conv2d

__all__ = [
    "GAPS",
    "Verdict",
    "catalog",
    "lookup",
    "lookup_torch",
]

# Coverage genuinely has gaps (unlike softmax's). Worded from what resolve()
# actually refuses, not aspirational -- see docs/content/kernels/convolutions.md.
GAPS: tuple[str, ...] = (
    "kernel_size not in {1, 3}: no kernel implements any other kernel size.",
    "dilation != 1: no kernel supports dilated convolution.",
    "groups not in {1, in_channels}: only plain (groups=1) and depthwise "
    "(groups=in_channels) convolution are implemented; grouped convolution "
    "with 1 < groups < in_channels has no kernel.",
    "standard (groups=1) conv at width > 128, other than exactly width=384 "
    "(conv_universal_wide384) or the fixed first-layer shape "
    "(conv_first_layer): no kernel covers it.",
    "stride=2 depthwise conv at a height/width combination outside "
    "{width=128; width in (16,32,64) with height a multiple of "
    "4*(128//width); height=width=16}: no kernel covers it.",
)


def lookup(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    has_bias: bool = False,
    apply_relu: bool = False,
    height: int,
    width: int,
) -> Verdict:
    """Return the kernel that computes this conv2d, or why none does.

    Mirrors :class:`~ipu_apps.convolutions_universal.layers.Conv2dDescription`'s
    fields directly (see its docstring for ``apply_relu``'s independence from
    ``has_bias``); this is just a keyword-argument-only convenience wrapper
    that resolves immediately instead of requiring the caller to build the
    dataclass first.
    """
    desc = Conv2dDescription(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, groups=groups, has_bias=has_bias,
        apply_relu=apply_relu, height=height, width=width,
    )
    return resolve("conv2d", **desc.params())


def lookup_torch(layer, input_shape, *, apply_relu: bool = False) -> Verdict:
    """Answer a query phrased the way ``torch.nn.Conv2d`` is called.

    Args:
        layer:       A real ``torch.nn.Conv2d`` instance.
        input_shape: ``[in_channels, height, width]`` (no batch dimension).
        apply_relu:  Must be passed explicitly -- a plain ``Conv2d`` has no
            activation concept; see :class:`Conv2dDescription`'s docstring.

    Returns:
        A :class:`~ipu_apps.kernel_registry.Verdict`.
    """
    desc = from_torch_conv2d(layer, input_shape, apply_relu=apply_relu)
    return resolve("conv2d", **desc.params())


def catalog() -> str:
    """Render the coverage table, probed from the kernels themselves."""
    lines = [
        "conv2d app coverage (all wide-vector FP32 mode)",
        "",
        "standard conv (groups=1, kernel_size=3, stride=1) -- by width, height=64",
    ]
    for b in boundaries(
        "conv2d", "width", range(1, 130),
        in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
        dilation=1, groups=1, has_bias=False, apply_relu=False, height=64,
    ):
        lines.append("  " + b.render("width"))

    lines += [
        "",
        "pointwise conv (groups=1, kernel_size=1) -- by width, height=64",
    ]
    for b in boundaries(
        "conv2d", "width", range(1, 130),
        in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0,
        dilation=1, groups=1, has_bias=False, apply_relu=False, height=64,
    ):
        lines.append("  " + b.render("width"))

    lines += [
        "",
        "depthwise conv, stride=1 (groups=in_channels, kernel_size=3) -- by width, height=64",
    ]
    for b in boundaries(
        "conv2d", "width", range(1, 130),
        in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
        dilation=1, groups=8, has_bias=False, apply_relu=False, height=64,
    ):
        lines.append("  " + b.render("width"))

    lines += [
        "",
        "depthwise conv, stride=2 (groups=in_channels, kernel_size=3) -- by width, height=128",
    ]
    for b in boundaries(
        "conv2d", "width", range(1, 130),
        in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1,
        dilation=1, groups=8, has_bias=False, apply_relu=False, height=128,
    ):
        lines.append("  " + b.render("width"))

    lines += [
        "",
        "depthwise conv, stride=2, width=16 -- by height",
    ]
    for b in boundaries(
        "conv2d", "height", range(1, 65),
        in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1,
        dilation=1, groups=8, has_bias=False, apply_relu=False, width=16,
    ):
        lines.append("  " + b.render("height"))

    lines += ["", "Open gaps:"]
    lines += [f"  - {g}" for g in GAPS]
    lines += [
        "",
        f"Kernels registered: {', '.join(k.name for k in kernels('conv2d'))}",
    ]
    return "\n".join(lines)
