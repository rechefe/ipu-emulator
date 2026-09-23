"""Shared code for the universal FP32 conv2d kernels.

The "universal" kernels -- ``conv_universal``, ``conv_universal_bn_activation``,
``conv_universal_wide384``, ``conv_first_layer``, ``depthwise_conv_universal``,
``depthwise_conv_universal_bn_activation``, the three
``depthwise_conv_stride2_*`` kernels and the two ``pointwise_conv_unified*``
kernels -- all run on the wide-vector FP32 datapath and answer the same
``conv2d`` query, so the vocabulary, the host-side packing helpers and the
SPEC boilerplate live here once.

**Query vocabulary.** Every universal kernel receives exactly the fields of
:class:`ConvQuery`, all required (``REQUIRES``):

``in_channels``, ``out_channels``   channel counts
``kernel_size``                     1 (pointwise) or 3 (spatial)
``stride``, ``padding``, ``dilation``
``groups``                          1 (plain), or ``in_channels`` (depthwise)
``has_bias``, ``apply_relu``        bias/activation, independent of each other
``height``, ``width``               spatial extent of the input

This is a different vocabulary from the older ``conv1x1`` / ``conv3x3_relu*``
kernels in this family (``shape`` / ``out_channels`` / ``kernel_size`` /
``stride`` / ``padding`` / ``activation``). The two coexist because each
kernel's ``requires`` refuses a query that lacks its parameters: a query in one
vocabulary is only ever claimed by the kernels speaking it.

**File layout.** Every universal kernel reads and writes raw, unpadded
``[channels, height, width]`` float32 (``input_path`` / ``output_path``);
packing into the on-device chunk-interleaved layout is internal to each
harness. Weights are ``kernel_path`` -- the torch ``[out, in / groups, k, k]``
weight tensor as raw float32 -- and, for the bias-capable kernels, ``bias_path``
holds ``[out_channels]`` float32.

**Framework adapter.** ``register_layer("Conv2d")`` lives here and registers on
import; every universal kernel's ``app.py`` imports this module, so
:func:`~ipu_apps.kernel_registry.lookup_layer` sees it once discovery ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np

from ipu_emu.ipu_config import DEFAULT_VALID_ELEMENTS

from ipu_apps.kernel_registry import (
    BIAS,
    INPUT,
    OUTPUT,
    WEIGHT,
    ExecutionConfig,
    ShapeBundle,
    UnsupportedLayer,
    folder_spec,
    no,
    register_layer,
    yes,
)

OP = "conv2d"
FAMILY_PACKAGE = "ipu_apps.kernels.convolutions"

# -- Shared constants (derived from IPU architecture parameters) -------------

# One chunk (= one XMEM row) holds DEFAULT_VALID_ELEMENTS lanes; in the FP32
# wide-vector mode these kernels run in, that is 512 bytes.
CHUNK_BYTES = DEFAULT_VALID_ELEMENTS                 # 128
CHUNK_ELEMENTS = CHUNK_BYTES                         # 128 elements/chunk
ROW_BYTES = CHUNK_ELEMENTS * 4                       # 512 B/row, FP32

WIDE_VECTOR_ONLY = (
    "FP32 wide-vector debug mode only (wide_vector_debug=True). This "
    "kernel has no INT8/quantized variant."
)


def kernel_asm(kernel: str) -> str:
    """Source of the family's ``<kernel>/<kernel>.asm`` (for multi-stage kernels).

    Every kernel target gets its family's ``.asm`` files as data, so a kernel
    may assemble a sibling's program as one of its stages.
    """
    return files(f"{FAMILY_PACKAGE}.{kernel}").joinpath(f"{kernel}.asm").read_text()


def render_asm(text: str, **context: Any) -> str:
    """Render a Jinja-templated ``.asm`` with ``context``, sandboxed.

    The assembler renders templates without a context (see ``ipu_as.template``
    for why rendering is sandboxed); kernels whose program is specialised per
    shape render it here first.
    """
    from jinja2.sandbox import SandboxedEnvironment

    return SandboxedEnvironment().from_string(text).render(**context)


def read_bias(bias, bias_path, count: int) -> np.ndarray:
    """``bias`` (array) or ``bias_path`` (raw float32 file); zeros if neither."""
    if bias is not None and bias_path is not None:
        raise ValueError("Provide at most one of bias= or bias_path=")
    if bias is None and bias_path is not None:
        bias = np.fromfile(bias_path, dtype="<f4")
    if bias is None:
        bias = np.zeros(count, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if bias.shape != (count,):
        raise ValueError(f"bias must have shape ({count},), got {bias.shape}")
    return bias


# -- Shared chunk-interleaved packing (channel-per-128-element-chunk layout) -
#
# Every FP32 wide-vector app in this family uses the same on-device layout
# for a [channels, rows, cols] tensor: rows_per_chunk = CHUNK_ELEMENTS // cols
# spatial rows share one 128-element (512-byte FP32) chunk, chunks are
# grouped by channel. This is strictly internal plumbing -- no app's
# input_path/output_path files are ever in this chunked format.


def pack_input_chunked(input_chw, cols: int) -> bytes:
    """Pack ``[channels, rows, cols]`` float32 into the chunk-interleaved
    layout every conv/depthwise/pointwise app in this family uses internally.

    ``cols`` must divide ``CHUNK_ELEMENTS`` (128). Offset formula (in
    ELEMENTS, not bytes): chunk = r // rows_per_chunk; local_row = r %
    rows_per_chunk; offset = (chunk*channels + ch)*128 + local_row*cols + c.
    """
    channels, rows, w = input_chw.shape
    if w != cols:
        raise ValueError(f"input_chw width {w} does not match cols {cols}")
    if CHUNK_ELEMENTS % cols != 0:
        raise ValueError(
            f"cols must divide {CHUNK_ELEMENTS} (a spatial row must tile one "
            f"chunk without straddling its edge), got {cols}"
        )
    rows_per_chunk = CHUNK_ELEMENTS // cols
    num_chunks = (rows + rows_per_chunk - 1) // rows_per_chunk
    packed = np.zeros(num_chunks * channels * CHUNK_ELEMENTS, dtype=np.float32)
    for ch in range(channels):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            row_off = (chunk * channels + ch) * CHUNK_ELEMENTS + local_row * cols
            packed[row_off:row_off + cols] = input_chw[ch, r, :].astype(np.float32)
    return packed.tobytes()


def unpack_output_chunked(raw: bytes, out_channels: int, out_rows: int, cols: int):
    """Inverse of :func:`pack_input_chunked`. Returns ``[out_channels,
    out_rows, cols]`` float32."""
    rows_per_chunk = CHUNK_ELEMENTS // cols
    out = np.zeros((out_channels, out_rows, cols), dtype=np.float32)
    arr = np.frombuffer(raw, dtype=np.float32)
    for ch in range(out_channels):
        for r in range(out_rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            off = (chunk * out_channels + ch) * CHUNK_ELEMENTS + local_row * cols
            out[ch, r, :] = arr[off:off + cols]
    return out


# -- Dynamic XMEM region layout ----------------------------------------------

# Region budget, in XMEM rows of CHUNK_ELEMENTS elements. Region sizes below
# are given in ELEMENTS, so the byte figure is the narrow-mode equivalent of
# this row budget (16384 rows).
XMEM_ROWS = 16384
XMEM_BYTES = XMEM_ROWS * CHUNK_BYTES  # 2 MiB


class XmemOverflow(ValueError):
    """Raised when an app's regions cannot fit inside XMEM.

    Carries the per-region breakdown so callers can see *which* region is
    responsible rather than only that the total is too large.
    """


def allocate_regions(regions, *, xmem_bytes: int = XMEM_BYTES) -> dict:
    """Pack named regions into non-overlapping, chunk-aligned bases.

    ``regions`` is a sequence of ``(name, size)`` pairs (sizes in elements),
    laid out in the order given starting at 0. Each base is rounded up to a
    ``CHUNK_BYTES`` boundary, because every XMEM operand in the row-addressed
    ISA is a ROW number -- a region starting mid-chunk is not addressable.

    Returns ``{name: base}``. Sizing from the actual configuration replaces
    per-app hardcoded ``*_BASE_ADDR`` gaps, which are silently wrong in both
    directions (a region overruns its successor at large configurations).

    Raises :class:`XmemOverflow` if the regions do not fit.
    """
    bases: dict = {}
    cursor = 0
    for name, size in regions:
        if size < 0:
            raise ValueError(f"region {name!r} has negative size {size}")
        bases[name] = cursor
        cursor += size
        cursor = ((cursor + CHUNK_BYTES - 1) // CHUNK_BYTES) * CHUNK_BYTES
    if cursor > xmem_bytes:
        detail = ", ".join(f"{n}={sz} B @ {bases[n]:#x}" for n, sz in regions)
        raise XmemOverflow(
            f"regions need {cursor} bytes ({cursor / 1048576:.2f} MiB) but XMEM "
            f"holds {xmem_bytes} ({xmem_bytes / 1048576:.2f} MiB): {detail}"
        )
    return bases


# -- Border mask (3x3 stride-1 kernels) --------------------------------------

# Mask slot assignment -- a single R_MASK blob (loaded once at init) carries all
# three slots the asm needs. Left/right edge columns are applied by mask_shift,
# NOT by slots; the slots only zero whole out-of-bounds rows:
#   0 = none        (KEEP all)             -> interior / kr=0-row taps
#   3 = top-row     (zero packed row 0)    -> g0 section kr=-1 taps
#   6 = bottom-row  (zero last packed row) -> gN section kr=+1 taps
MASK_SLOT_NONE = 0
MASK_SLOT_TOP = 3
MASK_SLOT_BOTTOM = 6


def build_border_mask_blob(cols: int) -> bytes:
    """Build the single 128-byte (8 x 16-byte slot) R_MASK blob.

    Mask polarity: a mask bit of **1 KEEPS** the lane, **0 ZEROES** it.
    ``rows_per_chunk`` = 128 // cols spatial rows are packed into the 128
    lanes; row ``r`` occupies lanes ``[r*cols, r*cols + cols)``. This blob is
    a bitmask, independent of the active arithmetic mode -- it does NOT widen
    with FP32/element size.

    Left/right edge columns are handled at runtime by ``mask_shift`` (with
    ``CR15.partition = cols``), so the slots only zero whole out-of-bounds rows:

      slot 0 (none)       -> KEEP every lane (interior / kr=0-row taps)
      slot 3 (top row)    -> ZERO packed row 0       (g0 section kr=-1 taps)
      slot 6 (bottom row) -> ZERO the last packed row (gN section kr=+1 taps)

    One blob carries all three; the asm selects slot 3 in g0 and slot 6 in gN,
    so no mid-program R_MASK reload is needed. Shared by conv_universal,
    conv_universal_bn_activation and both depthwise_conv_universal kernels.
    """
    rows_per_chunk = 128 // cols
    top_bits = set(range(0, cols))                                  # row 0
    bottom_row = rows_per_chunk - 1
    bottom_bits = set(range(bottom_row * cols, bottom_row * cols + cols))

    # Per slot, the set of lanes to ZERO (bit cleared); all others kept (bit 1).
    zero_lanes = {
        MASK_SLOT_NONE: set(),
        MASK_SLOT_TOP: top_bits,
        MASK_SLOT_BOTTOM: bottom_bits,
    }

    mask = bytearray(128)
    for slot, zeros in zero_lanes.items():
        for bit in range(128):
            if bit not in zeros:
                byte_idx = slot * 16 + bit // 8
                mask[byte_idx] |= 1 << (bit % 8)
    return bytes(mask)


# -- Query vocabulary --------------------------------------------------------

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
    every bias-capable app in this family unconditionally applies ReLU, so
    ``has_bias=True, apply_relu=False`` is refused by every universal kernel.
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
    def out_height(self) -> int:
        span = self.dilation * (self.kernel_size - 1) + 1
        return (self.height + 2 * self.padding - span) // self.stride + 1

    @property
    def out_width(self) -> int:
        span = self.dilation * (self.kernel_size - 1) + 1
        return (self.width + 2 * self.padding - span) // self.stride + 1

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
            derived={OUTPUT: (self.out_channels, self.out_height, self.out_width)},
        )


def conv_query(**params) -> ConvQuery:
    """Build a :class:`ConvQuery` from the registry's ``**params``.

    Kept as a function so every kernel indexes ``params`` the same way -- a
    typo'd key name fails identically everywhere instead of drifting per kernel.
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

    Every bias-capable app in this family (the ``_bn_activation`` twins)
    unconditionally applies ReLU, so this refusal is shared across all of them.
    """
    if q.has_bias and not q.apply_relu:
        return (
            "has_bias=True with apply_relu=False has no matching app: every "
            "bias-capable kernel in this family unconditionally applies ReLU"
        )
    return None


# -- Padding math (pure, no app-class dependency) -----------------------------

_VALID_COLS = (16, 32, 64, 128)


def next_valid_cols(width: int) -> int:
    """Smallest value in ``_VALID_COLS`` (k=3 apps) that is >= ``width``."""
    for cols in _VALID_COLS:
        if width <= cols:
            return cols
    raise ValueError(
        f"width {width} exceeds 128, the largest cols value any k=3 app in "
        "this family supports (conv_universal_wide384 handles wider images "
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
    number of 128-element chunks. Pointwise has no spatial neighbourhood, so
    a padded lane can never leak into a real lane through the conv.

    ``width`` must be <= 128: no integer greater than 128 divides 128, so the
    search for ``padded_cols`` has no terminating value above it (callers
    must refuse width > 128 in their own ``supports`` before reaching here;
    see ``pointwise_conv_unified``'s width bound)."""
    if width > 128:
        raise ValueError(f"width ({width}) must be <= 128; no divisor of 128 exists above it")
    padded_cols = width
    while 128 % padded_cols != 0:
        padded_cols += 1
    padded_rows = rows
    while (padded_rows * padded_cols) % 128 != 0:
        padded_rows += 1
    return padded_rows, padded_cols


def padding_caveat(q: ConvQuery, rows: int, cols: int) -> tuple[str, ...]:
    """The idle-lane caveat when ``(height, width)`` pads to ``(rows, cols)``."""
    if (rows, cols) == (q.height, q.width):
        return ()
    real = q.height * q.width
    padded = rows * cols
    return (
        f"{q.height}x{q.width} pads to {rows}x{cols}, so "
        f"{padded - real} of every {padded} spatial positions idle "
        f"({real / padded:.0%} utilisation).",
    )


# -- Spec helper --------------------------------------------------------------


def universal_spec(app_class, *, variant, supports, build, explain,
                   caveats=lambda q: (), cost=lambda q: 0.0):
    """KernelSpec for a universal conv2d kernel; callbacks receive the ConvQuery.

    ``supports(q)`` returns a refusal reason, or None to accept; positive
    extents are checked first. ``build(q)`` returns the harness kwargs.
    ``caveats(q)`` adds to the shared FP32-only caveat.
    """
    def query(params):
        return conv_query(**params)

    def _supports(**params):
        q = query(params)
        reason = positive_dims(q) or supports(q)
        return no(reason) if reason else yes()

    return folder_spec(
        app_class,
        op=OP,
        variant=variant,
        # Every callback indexes these, so the registry checks them first: a
        # query in another conv2d vocabulary is refused, naming what is missing.
        requires=REQUIRES,
        tags=("fp32-wide",),
        supports=_supports,
        build=lambda **params: build(query(params)),
        explain=lambda **params: explain(query(params)),
        caveats=lambda **params: (WIDE_VECTOR_ONLY, *caveats(query(params))),
        bundle=lambda **params: query(params).bundle,
        cost=lambda **params: cost(query(params)),
        execution=ExecutionConfig(mode="fp32"),
    )


# -- Framework-layer adapter ---------------------------------------------------


@dataclass(frozen=True)
class Conv2dDescription:
    """Framework-free description of a ``Conv2d`` query (the conv2d params).

    Mirrors the fields of ``torch.nn.Conv2d`` that matter for dispatch; build
    one directly (no torch needed) or via :func:`from_torch_conv2d`.
    ``apply_relu`` must be requested explicitly (see :class:`ConvQuery`).
    """

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

    def params(self) -> dict:
        return {
            "in_channels": self.in_channels, "out_channels": self.out_channels,
            "kernel_size": self.kernel_size, "stride": self.stride,
            "padding": self.padding, "dilation": self.dilation,
            "groups": self.groups, "has_bias": self.has_bias,
            "apply_relu": self.apply_relu, "height": self.height,
            "width": self.width,
        }


def from_torch_conv2d(layer, input_shape=None, *, apply_relu: bool = False) -> Conv2dDescription:
    """Build a :class:`Conv2dDescription` from a ``torch.nn.Conv2d``-like layer.

    Matched by attributes, so torch is never imported. ``input_shape``, if
    given, is ``(in_channels, height, width)``; height and width default to 0
    (deliberately unresolvable) when omitted.

    Raises:
        UnsupportedLayer: for configurations no kernel models -- non-square
            parameters, string padding (``"same"``/``"valid"``), a padding
            mode other than zeros, or an input shape that is not
            ``(in_channels, height, width)``.
    """
    names = ("in_channels", "out_channels", "kernel_size", "stride", "padding",
             "dilation", "groups", "bias")
    missing = [n for n in names if not hasattr(layer, n)]
    if missing:
        raise UnsupportedLayer(
            f"{type(layer).__name__} is missing expected attribute(s) "
            f"{', '.join(missing)}; it does not look like the layer this "
            f"adapter was written for"
        )

    def _one(name) -> int:
        v = getattr(layer, name)
        if isinstance(v, str):
            raise UnsupportedLayer(
                f"{name}={v!r} is not supported; give an explicit integer {name}"
            )
        if isinstance(v, (tuple, list)):
            if len(set(v)) != 1:
                raise UnsupportedLayer(
                    f"non-square parameter {name}={tuple(v)} is not supported "
                    "(kernel_size, stride, padding, and dilation must be "
                    "scalars or square tuples)"
                )
            return int(v[0])
        return int(v)

    padding_mode = getattr(layer, "padding_mode", "zeros")
    if padding_mode != "zeros":
        raise UnsupportedLayer(
            f"padding_mode={padding_mode!r} is not supported; every kernel "
            "zero-pads"
        )

    height, width = (0, 0)
    if input_shape is not None:
        shape = tuple(int(d) for d in input_shape)
        if len(shape) != 3:
            raise UnsupportedLayer(
                f"input_shape must be (in_channels, height, width) -- these "
                f"kernels process one image, no batch dimension; got {shape}"
            )
        if shape[0] != int(layer.in_channels):
            raise UnsupportedLayer(
                f"input_shape has {shape[0]} channels but the layer expects "
                f"in_channels={layer.in_channels}"
            )
        _, height, width = shape

    return Conv2dDescription(
        in_channels=int(layer.in_channels),
        out_channels=int(layer.out_channels),
        kernel_size=_one("kernel_size"),
        stride=_one("stride"),
        padding=_one("padding"),
        dilation=_one("dilation"),
        groups=int(layer.groups),
        has_bias=layer.bias is not None,
        apply_relu=apply_relu,
        height=height,
        width=width,
    )


@register_layer("Conv2d")
def _conv2d_layer(layer, input_shape):
    """``nn.Conv2d`` -> the ``conv2d`` operation, in the universal vocabulary.

    ``apply_relu`` has no ``torch.nn.Conv2d`` equivalent -- routed here as
    ``False``, the only value derivable from the layer alone. A caller that
    wants the ReLU twin builds a :class:`Conv2dDescription` via
    ``from_torch_conv2d(layer, input_shape, apply_relu=True)`` and resolves
    ``resolve("conv2d", **desc.params())`` directly.
    """
    return OP, from_torch_conv2d(layer, input_shape, apply_relu=False).params()
