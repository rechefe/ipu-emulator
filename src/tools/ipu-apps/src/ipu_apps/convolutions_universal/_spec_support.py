"""Shared helpers for the convolutions_universal kernels' registry declarations.

Every kernel in this package answers the same query -- a ``Conv2d``-shaped
description plus the input's spatial shape -- so the parameter vocabulary and
the shared refusal reasons live here rather than being repeated across ten
``__init__.py`` files.

The query parameters a convolution kernel receives are exactly the fields of
:class:`ConvQuery`, all required:

``in_channels``, ``out_channels``   channel counts
``kernel_size``                     1 (pointwise) or 3 (spatial)
``stride``, ``padding``, ``dilation``
``groups``                          1 (plain), or ``in_channels`` (depthwise)
``has_bias``, ``apply_relu``        bias/activation, independent of each other
                                     (see :class:`ConvQuery`'s docstring)
``height``, ``width``               spatial extent of the input this instance
                                     will actually run against

Height/width are part of the query (unlike a framework layer's own
configuration) because the underlying apps' region layout, row-count floors,
and column-alignment constraints are all shape-dependent -- a kernel that
handles a 3x3 depthwise conv at 32x32 may refuse the identical configuration
at 3x200 (width > 128, no padding path in this module handles it).

This module intentionally has NO opinion on padding/packing/dispatch --
that plumbing (``run_layer``, ``pack_input_chunked``, ``_pointwise_pad_shape``,
...) stays in :mod:`ipu_apps.convolutions_universal.layers`, which is now a
thin wrapper that builds a :class:`ConvQuery` and delegates the "does
anything cover this" question to the registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ipu_apps.kernel_registry import BIAS, INPUT, OUTPUT, WEIGHT, ShapeBundle

REQUIRES = (
    "in_channels", "out_channels", "kernel_size", "stride", "padding",
    "dilation", "groups", "has_bias", "apply_relu", "height", "width",
)


@dataclass(frozen=True)
class ConvQuery:
    """A convolution query, exactly mirroring ``torch.nn.Conv2d``'s shape.

    ``apply_relu`` has no ``torch.nn.Conv2d`` equivalent -- a plain Conv2d
    layer never implies an activation, so callers must state it explicitly
    rather than have it inferred from ``has_bias``. The two are independent:
    every app with bias support in this package unconditionally applies ReLU,
    so ``has_bias=True, apply_relu=False`` is refused by every kernel.
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dilation: int
    groups: int
    has_bias: bool
    apply_relu: bool
    height: int
    width: int

    @property
    def is_depthwise(self) -> bool:
        return self.groups == self.in_channels

    @property
    def bundle(self) -> ShapeBundle:
        return ShapeBundle.of(
            **{
                INPUT: (self.in_channels, self.height, self.width),
                WEIGHT: (
                    self.out_channels, self.in_channels // self.groups,
                    self.kernel_size, self.kernel_size,
                ),
                **({BIAS: (self.out_channels,)} if self.has_bias else {}),
            }
        ).with_shapes(
            derived={OUTPUT: (self.out_channels, self.height, self.width)},
        )


def conv_query(**params) -> ConvQuery:
    """Build a :class:`ConvQuery` from the registry's ``**params``.

    Kept as a function (rather than constructing ``ConvQuery`` directly in
    every ``supports``/``build``) so every kernel indexes ``params`` the same
    way -- a typo'd key name fails identically everywhere instead of drifting
    per kernel.
    """
    return ConvQuery(
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
        kernel_size=int(params["kernel_size"]),
        stride=int(params["stride"]),
        padding=int(params["padding"]),
        dilation=int(params["dilation"]),
        groups=int(params["groups"]),
        has_bias=bool(params["has_bias"]),
        apply_relu=bool(params["apply_relu"]),
        height=int(params["height"]),
        width=int(params["width"]),
    )


def positive_dims(q: ConvQuery) -> str | None:
    """Return a refusal reason if the problem has a non-positive extent."""
    if q.height < 1:
        return f"height ({q.height}) must be >= 1"
    if q.width < 1:
        return f"width ({q.width}) must be >= 1"
    if q.in_channels < 1:
        return f"in_channels ({q.in_channels}) must be >= 1"
    if q.out_channels < 1:
        return f"out_channels ({q.out_channels}) must be >= 1"
    return None


def bias_requires_relu(q: ConvQuery) -> str | None:
    """Return a refusal reason if bias is requested without ReLU.

    Every bias-capable app in this package (the ``_bn_activation`` twins)
    unconditionally applies ReLU, so this refusal is shared across all of
    them rather than restated per kernel.
    """
    if q.has_bias and not q.apply_relu:
        return (
            "has_bias=True with apply_relu=False has no matching app: every "
            "bias-capable kernel in this package unconditionally applies ReLU"
        )
    return None


# -- Padding math (pure, no app-class dependency) -----------------------------
#
# Lives here rather than in layers.py so a kernel's own SPEC.build (which
# needs the padded rows/cols an app constructor actually takes) and
# layers.py's data-plumbing (which needs the same numbers to pad/pack the
# real tensor) can both import it without either module importing the other.

_VALID_COLS = (16, 32, 64, 128)


def next_valid_cols(width: int) -> int:
    """Smallest value in ``_VALID_COLS`` (k=3 apps) that is >= ``width``."""
    for cols in _VALID_COLS:
        if width <= cols:
            return cols
    raise ValueError(
        f"width {width} exceeds 128, the largest cols value any k=3 app in "
        "this package supports (conv_universal_wide384 handles wider images "
        "but is experimental/unoptimized and not part of this dispatch)"
    )


def min_rows_for_chunk_floor(rows: int, cols: int) -> int:
    """Smallest ``padded_rows >= rows`` such that ``padded_rows * cols >= 256``
    (the ``num_chunks >= 2`` floor every stride-1 k=3 app enforces)."""
    needed = math.ceil(256 / cols)
    return max(rows, needed)


def pointwise_pad_shape(rows: int, width: int) -> tuple[int, int]:
    """Smallest ``(padded_rows, padded_cols)`` >= ``(rows, width)`` such that
    ``padded_cols`` divides 128 and ``padded_rows * padded_cols`` is a whole
    number of 128-byte chunks. See ``layers.py``'s original docstring for the
    pointwise-specific reasoning (no spatial neighbourhood, so no mask care
    is needed for the padded lanes)."""
    padded_cols = width
    while 128 % padded_cols != 0:
        padded_cols += 1
    padded_rows = rows
    while (padded_rows * padded_cols) % 128 != 0:
        padded_rows += 1
    return padded_rows, padded_cols


def stride2_padded_rows(height: int) -> int:
    """Smallest ``padded_rows >= height`` that is a multiple of 4 and >= 4
    (``depthwise_conv_stride2_128``'s own row-count constraint)."""
    padded_rows = height if height % 4 == 0 else height + (4 - height % 4)
    return max(padded_rows, 4)
