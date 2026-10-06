"""Shared code for the convolution kernels: harness bases, host-side packing and SPEC helpers.

The family speaks two ``conv2d`` query vocabularies. They coexist because each
kernel's ``requires`` refuses a query that lacks its parameters: a query in one
vocabulary is only ever claimed by the kernels speaking it.

**Memory-layout kernels** (``conv1x1``, ``conv3x3_relu``,
``conv3x3_relu_cin1``) subclass :class:`ConvApp` and take ``shape`` /
``out_channels`` / ``kernel_size`` / ``stride`` / ``padding`` /
``activation``. Their CR maps and row counts match the adjacent assembly
headers.

**Universal kernels** -- ``conv_universal``, ``conv_universal_bn_activation``,
``conv_universal_wide384``, ``conv_first_layer``, ``depthwise_conv_universal``,
``depthwise_conv_universal_bn_activation``, the three
``depthwise_conv_stride2_*`` kernels and the two ``pointwise_conv_unified*``
kernels -- run on the wide-vector FP32 datapath and receive exactly the fields
of :class:`ConvQuery`, all required (``REQUIRES``):

``in_channels``, ``out_channels``   channel counts
``kernel_size``                     1 (pointwise) or 3 (spatial)
``stride``, ``padding``, ``dilation``
``groups``                          1 (plain), or ``in_channels`` (depthwise)
``has_bias``, ``apply_relu``        bias/activation, independent of each other
``height``, ``width``               spatial extent of the input

Every universal kernel reads and writes raw, unpadded
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
    Table,
    UnsupportedLayer,
    folder_spec,
    no,
    register_layer,
    yes,
)
from ipu_apps.kernel_registry.memory import MemoryApp, MemoryLayout, ceildiv, positive_shape


# -- Memory-layout kernels ----------------------------------------------------


class ConvApp(MemoryApp):
    kernel_size = 1
    single_channel = False

    @classmethod
    def memory_layout(cls, *, shape, out_channels, kernel_size, stride, padding, activation):
        c, h, w = positive_shape(shape, 3)
        positive_shape((out_channels,), 1)
        k = cls.kernel_size
        if kernel_size != k or stride != 1 or padding != k // 2:
            raise ValueError(f"requires kernel_size={k}, stride=1, padding={k // 2}")
        expected_activation = "none" if k == 1 else "relu"
        if activation != expected_activation:
            raise ValueError(f"requires activation={expected_activation}")
        if cls.single_channel and c != 1:
            raise ValueError("requires one input channel")
        tiles = ceildiv(w, 128 if k == 1 else 126)
        groups = ceildiv(c, 128 if k == 1 else 14)
        weights = (c + int(not cls.single_channel)) * (h + k - 1) * tiles
        bias = weights + out_channels * groups
        output = bias + out_channels
        if cls.single_channel:
            crs = {2: 0, 3: output, 4: weights, 5: bias, 6: tiles,
                   7: h, 8: out_channels, 9: 126, 10: 128, 11: tiles - 1}
        else:
            crs = {2: 0, 3: output, 4: weights, 5: bias, 6: h * tiles,
                   7: tiles, 8: h, 9: c, 10: out_channels, 11: groups, 12: 128}
            if k == 3:
                crs.update({13: 13, 14: 126})
        return MemoryLayout(output, out_channels * h * tiles, crs)

    @classmethod
    def tables(cls, *, shape, out_channels, kernel_size, stride, padding, activation):
        """XMEM tables as the Cache Unit sees them (see ``ipu_emu.table_trace``).

        Input channel ``c`` is a plane of ``P = (h + k - 1) * tiles`` rows.
        Every output channel re-walks the whole input (one repetition each):
        per output tile it walks down the channels, so the next tile steps back
        from the last channel to the first. The multi-channel kernels' channel
        loop prefetches into the guard plane past the last channel -- rows
        that are allocated and declared as overhead.
        """
        c, h, w = shape
        k = cls.kernel_size
        tiles = ceildiv(w, 128 if k == 1 else 126)
        groups = ceildiv(c, 128 if k == 1 else 14)
        plane = (h + k - 1) * tiles
        if cls.single_channel:
            # Rows y, y+1, y+2 of one tile, then the next tile: 2*tiles - 1 back.
            data = Table(cr=2, rows=plane, repetitions=out_channels, jump_back=2 * tiles - 1)
        elif k == 1:
            # Channel Cin's row at each tile position: the whole guard plane.
            data = Table(cr=2, rows=c * plane, repetitions=out_channels,
                         jump_back=c * plane - 1, overhead_rows=plane)
        else:
            # Channel Cin's rows y and y+1 at each tile: guard rows up to (h+1)*tiles.
            data = Table(cr=2, rows=c * plane, repetitions=out_channels,
                         jump_back=c * plane + tiles - 1, overhead_rows=(h + 1) * tiles)
        return {
            "input": data,
            # An output channel's group rows are re-read for every tile.
            "weights": Table(cr=4, rows=out_channels * groups,
                             jump_back=groups - 1 if h * tiles > 1 else 0),
            "bias": Table(cr=5, rows=out_channels),
            "output": Table(cr=3, rows=out_channels * h * tiles),
        }


# -- Universal kernels: shared constants --------------------------------------

OP = "conv2d"
FAMILY_PACKAGE = "ipu_apps.kernels.convolutions"

# One chunk (= one XMEM row) holds DEFAULT_VALID_ELEMENTS lanes; in the FP32
# wide-vector mode these kernels run in, that is 512 bytes. CHUNK_BYTES is the
# narrow-mode byte count, numerically equal to the element count.
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
# are given in ELEMENTS; XMEM_BYTES is the narrow-mode byte figure of this row
# budget, numerically equal to its element count (16384 rows x 128 elements).
XMEM_ROWS = 16384
XMEM_BYTES = XMEM_ROWS * CHUNK_BYTES  # 2 Mi elements


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

    Returns ``{name: base}``, with each region sized from the actual
    configuration so that no region overruns its successor.

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
        detail = ", ".join(f"{n}={sz} elements @ {bases[n]:#x}" for n, sz in regions)
        raise XmemOverflow(
            f"regions need {cursor} elements ({cursor // CHUNK_ELEMENTS} XMEM rows) "
            f"but XMEM holds {xmem_bytes} ({xmem_bytes // CHUNK_ELEMENTS} rows): {detail}"
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
                   caveats=lambda q: (), cost=lambda q: 0.0, tables=None):
    """KernelSpec for a universal conv2d kernel; callbacks receive the ConvQuery.

    ``supports(q)`` returns a refusal reason, or None to accept; positive
    extents are checked first. ``build(q)`` returns the harness kwargs.
    ``caveats(q)`` adds to the shared FP32-only caveat. ``tables(q)``, if
    given, is the kernel's ``SPEC.tables`` declaration.
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
        tables=None if tables is None else (lambda **params: tables(query(params))),
    )


def standard3x3_tables(q: ConvQuery) -> dict[str, Table]:
    """XMEM tables of ``conv_universal`` and its ``_bn_activation`` twin.

    The image pads to ``rows x cols`` and packs into ``N`` 128-element chunks;
    chunk ``n`` of all ``C`` input channels is one group of ``C`` rows. The
    input table holds one group of headroom (chunk -1, the kr=-1 rows of
    chunk 0) and then the data. Chunk by chunk, every filter re-reads the
    kr=-1, 0, +1 groups -- ``3*C - 1`` rows back -- and the last chunk's kr=+1
    group lies one group past the table: zeroed by the bottom border mask, so
    relied upon, and declared as overhead. The weights (two rows per FPB=28
    super-block) are re-walked once per chunk.
    """
    cols = next_valid_cols(q.width)
    chunks = min_rows_for_chunk_floor(q.height, cols) * cols // CHUNK_ELEMENTS
    c = q.in_channels
    weight_rows = q.out_channels * math.ceil(c / 28) * 2
    return {
        "input": Table(cr=10, rows=(chunks + 1) * c, jump_back=3 * c - 1, overhead_rows=c),
        "weights": Table(cr=5, rows=weight_rows, repetitions=chunks),
        "mask": Table(cr=3, rows=1),
        "output": Table(cr=2, rows=chunks * q.out_channels),
    }


def depthwise3x3_tables(q: ConvQuery, channels_per_block: int) -> dict[str, Table]:
    """XMEM tables of ``depthwise_conv_universal`` and its ``_bn_activation`` twin.

    Same chunk layout as :func:`standard3x3_tables` with ``C`` channels: one
    group of headroom, then ``N`` chunk groups. Chunk by chunk, each channel
    reads its kr=-1, 0, +1 rows, ``2*C`` rows back at most; the last chunk's
    pipeline loads one row past the table. The weights (two rows per
    super-block of ``channels_per_block`` channels) are re-walked per chunk.

    The deferred-store pipeline drains one store before the first output, so
    CR2 is the output base minus one: the output table's first row is that
    dummy store's row -- in today's layout, the border-mask row (the mask is
    read before it is overwritten).
    """
    cols = next_valid_cols(q.width)
    chunks = min_rows_for_chunk_floor(q.height, cols) * cols // CHUNK_ELEMENTS
    c = q.in_channels
    return {
        "input": Table(cr=10, rows=(chunks + 1) * c, jump_back=2 * c, overhead_rows=1),
        "weights": Table(cr=5, rows=math.ceil(c / channels_per_block) * 2, repetitions=chunks),
        "mask": Table(cr=3, rows=1),
        "output": Table(cr=2, rows=1 + chunks * c),
    }


def pointwise_tables(q: ConvQuery, *, bias: bool = False) -> dict[str, Table]:
    """XMEM tables of ``pointwise_conv_unified`` and its ``_bn_activation`` twin.

    The image pads to ``G`` 128-element row groups; row group ``g`` holds one
    row per input channel. Every output channel re-walks its row group's
    ``C`` rows (``C - 1`` back), and the kernel -- one row per (output
    channel, 128-channel pass) -- is re-walked for every row group, as is the
    bias region that mirrors it in the twin.
    """
    rows, cols = pointwise_pad_shape(q.height, q.width)
    groups = rows * cols // CHUNK_ELEMENTS
    c = q.in_channels
    kernel_rows = q.out_channels * math.ceil(c / 128)
    tables = {
        "input": Table(cr=4, rows=groups * c, jump_back=c - 1),
        "kernel": Table(cr=14, rows=kernel_rows, repetitions=groups),
        "mask": Table(cr=2, rows=1),
        "output": Table(cr=3, rows=groups * q.out_channels),
    }
    if bias:
        tables["bias"] = Table(cr=10, rows=kernel_rows, repetitions=groups)
    return tables


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
