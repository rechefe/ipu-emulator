"""The ``Conv2d`` layer adapter (E06-16, E07-16): a ``torch.nn.Conv2d``-like
layer and its input shape as a ``conv2d`` query in the universal vocabulary.

``Registry.from_layer`` finds it through ``LAYER_ADAPTERS``. The layer is
matched by its attributes, so torch is never imported. :class:`Conv2dGeometry`
is the query's eleven fields (``ipu_apps``' ``ConvQuery`` and
``Conv2dDescription`` declared the same ones twice). ``apply_relu`` has no
``torch.nn.Conv2d`` equivalent: a layer alone never implies an activation, so
the adapter asks for none, and a caller that wants the ReLU kernels builds the
geometry with ``from_torch_conv2d(..., apply_relu=True)`` and resolves its
``params()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ipu_harness.registry import UnsupportedLayer

__all__ = ["LAYER_ADAPTERS", "Conv2dGeometry", "from_torch_conv2d"]


@dataclass(frozen=True)
class Conv2dGeometry:
    """A ``conv2d`` query in the universal vocabulary."""

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    has_bias: bool
    apply_relu: bool = False
    height: int = 0
    width: int = 0

    def params(self) -> dict[str, Any]:
        """The query's parameters, by name."""
        return asdict(self)


_ATTRIBUTES = (
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
    "bias",
)


def _square(layer: object, name: str) -> int:
    value = getattr(layer, name)
    if isinstance(value, str):
        raise UnsupportedLayer(
            f"{name}={value!r} is not supported; give an explicit integer {name}"
        )
    if isinstance(value, (tuple, list)):
        if len(set(value)) != 1:
            raise UnsupportedLayer(
                f"non-square parameter {name}={tuple(value)} is not supported "
                "(kernel_size, stride, padding, and dilation must be scalars or square tuples)"
            )
        return int(value[0])
    return int(value)


def from_torch_conv2d(
    layer: Any, input_shape: tuple[int, ...] | None = None, *, apply_relu: bool = False
) -> Conv2dGeometry:
    """The geometry of ``layer`` on an input of ``(in_channels, height, width)``.

    Height and width are 0 (no kernel accepts them) when ``input_shape`` is
    left out.

    Raises:
        UnsupportedLayer: A configuration no kernel models: a missing
            attribute, a non-square parameter, string padding, a padding mode
            other than zeros, or an input shape that is not
            ``(in_channels, height, width)``.
    """
    missing = [n for n in _ATTRIBUTES if not hasattr(layer, n)]
    if missing:
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) {', '.join(missing)}; "
            "it does not look like the layer this adapter was written for"
        )
    padding_mode = getattr(layer, "padding_mode", "zeros")
    if padding_mode != "zeros":
        raise UnsupportedLayer(
            f"padding_mode={padding_mode!r} is not supported; every kernel zero-pads"
        )
    height = width = 0
    if input_shape is not None:
        shape = tuple(int(d) for d in input_shape)
        if len(shape) != 3:
            raise UnsupportedLayer(
                "input_shape must be (in_channels, height, width) -- these kernels "
                f"process one image, no batch dimension; got {shape}"
            )
        if shape[0] != int(layer.in_channels):
            raise UnsupportedLayer(
                f"input_shape has {shape[0]} channels but the layer expects "
                f"in_channels={layer.in_channels}"
            )
        _, height, width = shape
    return Conv2dGeometry(
        in_channels=int(layer.in_channels),
        out_channels=int(layer.out_channels),
        kernel_size=_square(layer, "kernel_size"),
        stride=_square(layer, "stride"),
        padding=_square(layer, "padding"),
        dilation=_square(layer, "dilation"),
        groups=int(layer.groups),
        has_bias=layer.bias is not None,
        apply_relu=apply_relu,
        height=height,
        width=width,
    )


def _conv2d(layer: Any, shape: tuple[int, ...]) -> tuple[str, dict[str, Any]]:
    return "conv2d", from_torch_conv2d(layer, shape).params()


LAYER_ADAPTERS = {"Conv2d": _conv2d}
