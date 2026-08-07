"""Standalone PyTorch ``nn.Conv2d`` adapter for the convolutions_universal apps.

Runs a real ``torch.nn.Conv2d`` module (or a framework-free equivalent
description) against the IPU emulator by dispatching to whichever existing
app class matches the layer's ``groups``/``stride``/``bias`` combination,
handling all the data plumbing (INT8 quantization, chunk-interleaved packing,
width padding to a supported ``cols`` value, and output unpacking) that the
apps themselves don't provide.

This module is intentionally NOT wired into ``ipu_apps.kernel_registry`` --
that registry (kernels self-declare a ``SPEC``, discovery walks the app tree)
is still an open, unmerged PR (#206) with convolution registration explicitly
out of its scope. ``run_layer``/``resolve`` here are meant to be trivially
re-exposed as one more ``SPEC`` once that registry lands; until then this
module has no registry dependency and works standalone.

Supported today:
  - ``kernel_size == 3``, ``dilation == 1``, zero-padding == 1 ("same" for k=3)
  - ``stride in {1, 2}``
  - ``groups in {1, in_channels}`` (plain conv or fully depthwise)
  - ``dtype`` INT8 only (the only dtype the underlying apps' bias/activation
    path supports end-to-end)
  - spatial width (and height, for the depthwise stride-2 path) up to 128,
    via zero-padding up to the nearest supported ``cols`` and truncating the
    output back to the true width -- verified bit-exact against a numpy
    reference at width 65/99/100/101/125/126/127 for both stride 1 and 2
    (see the throwaway probe this module promotes into real, tested code).

Explicitly out of scope (raises :class:`UnsupportedLayer`):
  - ``dilation != 1``, ``kernel_size != 3``, non-"same" padding
  - ``groups`` not in ``{1, in_channels}`` (grouped-but-not-depthwise conv)
  - ``stride`` not in ``{1, 2}``
  - width > 128 (nothing but the experimental, unoptimized
    ``conv_universal_wide384`` handles that, and it is not wired in here)
  - folding a separate ``nn.BatchNorm2d`` -- only ``Conv2d.bias`` (a plain
    per-output-channel bias tensor) is supported; BN-folding math is a
    distinct concern from this module's I/O plumbing and is deferred.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from ipu_as.lark_tree import assemble_to_bin_file

from ipu_apps.convolutions_universal import CHUNK_BYTES
from ipu_apps.convolutions_universal.conv.conv_universal import (
    ConvUniversalApp,
    OUTPUT_CHUNK_BYTES as _CONV_OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.conv.conv_universal_bn_activation import (
    ConvUniversalBnActivationApp,
    OUTPUT_CHUNK_BYTES as _CONV_BN_OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal import (
    DepthwiseConvUniversalApp,
    OUTPUT_CHUNK_BYTES as _DW_OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_universal_bn_activation import (
    DepthwiseConvUniversalBnActivationApp,
    OUTPUT_CHUNK_BYTES as _DW_BN_OUTPUT_CHUNK_BYTES,
)
from ipu_apps.convolutions_universal.depthwise.depthwise_conv_stride2_128 import (
    DepthwiseConvStride2_128App,
)

if TYPE_CHECKING:
    import torch

# None of the app modules export their own ASM_PATH constant; derive them
# here the same way each app's own test file does.
_CONV_ASM_PATH = (
    Path(__file__).resolve().parent
    / "conv" / "conv_universal" / "conv_universal.asm"
)
_CONV_BN_ASM_PATH = (
    Path(__file__).resolve().parent
    / "conv" / "conv_universal_bn_activation" / "conv_universal_bn_activation.asm"
)
_DW_ASM_PATH = (
    Path(__file__).resolve().parent
    / "depthwise" / "depthwise_conv_universal" / "depthwise_conv_universal.asm"
)
_DW_BN_ASM_PATH = (
    Path(__file__).resolve().parent
    / "depthwise" / "depthwise_conv_universal_bn_activation"
    / "depthwise_conv_universal_bn_activation.asm"
)

_VALID_COLS = (16, 32, 64, 128)


class UnsupportedLayer(ValueError):
    """Raised when a layer/shape combination has no matching IPU app.

    One instance names exactly one violated constraint -- callers should not
    need to guess which of several possible reasons caused the refusal.
    """


# ============================================================================
# Width padding / truncation (verified against a numpy reference; see the
# module docstring and test_layers.py for the boundary-width checks this
# promotes into real code)
# ============================================================================

def _next_valid_cols(width: int) -> int:
    """Smallest value in ``_VALID_COLS`` that is >= ``width``."""
    for cols in _VALID_COLS:
        if width <= cols:
            return cols
    raise UnsupportedLayer(
        f"width {width} exceeds 128, the largest cols value any app in this "
        "module supports (conv_universal_wide384 handles wider images but is "
        "experimental/unoptimized and not wired into this adapter)"
    )


def pad_width_to_cols(input_chw: np.ndarray, cols: int) -> np.ndarray:
    """Zero-pad an ``[channels, rows, width]`` array's last axis up to ``cols``."""
    channels, rows, width = input_chw.shape
    if width == cols:
        return input_chw
    padded = np.zeros((channels, rows, cols), dtype=input_chw.dtype)
    padded[:, :, :width] = input_chw
    return padded


def truncate_width(output_cow: np.ndarray, true_out_width: int) -> np.ndarray:
    """Slice an ``[out_channels, out_rows, cols]`` array's last axis down to the
    true (unpadded) output width."""
    return output_cow[:, :, :true_out_width]


# ============================================================================
# INT8 quantization (refuses to guess a scale for you)
# ============================================================================

def quantize_int8(tensor: "torch.Tensor") -> np.ndarray:
    """Convert an already-integer-valued tensor to an INT8 numpy array.

    This module does not choose a quantization scale/zero-point on your
    behalf -- that is a modeling decision, not a plumbing one. Pass a tensor
    whose values are already integers in [-128, 127] (e.g. the output of your
    own quantization scheme, or a tensor built from an INT8 source). A tensor
    with genuine fractional values raises, rather than silently rounding.
    """
    arr = tensor.detach().cpu().numpy()
    rounded = np.round(arr)
    if not np.allclose(arr, rounded, atol=1e-6):
        raise UnsupportedLayer(
            "tensor has non-integer values; run_layer does not choose a "
            "quantization scale for you -- quantize to integers in "
            "[-128, 127] before calling (see module docstring)"
        )
    if rounded.min() < -128 or rounded.max() > 127:
        raise UnsupportedLayer(
            f"tensor values [{rounded.min()}, {rounded.max()}] exceed the "
            "INT8 range [-128, 127] after rounding"
        )
    return rounded.astype(np.int8)


# ============================================================================
# Chunk-interleaved packing (matches every app's own input layout)
# ============================================================================

def pack_input_chunked(input_chw: np.ndarray) -> bytes:
    """Pack ``[channels, rows, cols]`` int8 into the chunk-interleaved layout
    every app in this module expects from ``input_path``.

    ``cols`` must already be one of ``_VALID_COLS`` (pad first if not).
    Offset formula matches ``_pack_input_chunked`` in the existing conv/
    depthwise test files: chunk = r // rows_per_chunk; local_row = r %
    rows_per_chunk; offset = (chunk*channels + ch)*128 + local_row*cols + c.
    """
    channels, rows, cols = input_chw.shape
    if cols not in _VALID_COLS:
        raise ValueError(f"cols must be one of {_VALID_COLS}, got {cols}")
    rows_per_chunk = 128 // cols
    num_chunks = math.ceil(rows / rows_per_chunk)
    packed = bytearray(num_chunks * channels * 128)
    for ch in range(channels):
        for r in range(rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            row_off = (chunk * channels + ch) * 128 + local_row * cols
            packed[row_off:row_off + cols] = input_chw[ch, r, :].astype(np.int8).tobytes()
    return bytes(packed)


def unpack_output_chunked(
    raw: bytes, out_channels: int, out_rows: int, cols: int,
) -> np.ndarray:
    """Inverse of :func:`pack_input_chunked`, for an app's dumped output file.

    Returns ``[out_channels, out_rows, cols]`` int8.
    """
    rows_per_chunk = 128 // cols
    out = np.zeros((out_channels, out_rows, cols), dtype=np.int8)
    arr = np.frombuffer(raw, dtype=np.int8)
    for ch in range(out_channels):
        for r in range(out_rows):
            chunk = r // rows_per_chunk
            local_row = r % rows_per_chunk
            off = (chunk * out_channels + ch) * 128 + local_row * cols
            out[ch, r, :] = arr[off:off + cols]
    return out


# ============================================================================
# Layer description + validation
# ============================================================================

@dataclass(frozen=True)
class Conv2dDescription:
    """Framework-free description of what :func:`resolve` needs to know.

    Mirrors the fields of ``torch.nn.Conv2d`` that matter for dispatch; build
    one directly (no torch needed) or via :func:`from_torch_conv2d`.

    ``apply_relu`` has no ``torch.nn.Conv2d`` equivalent -- a plain Conv2d
    layer never implies an activation, so this must be requested explicitly
    (see :func:`run_layer`'s ``apply_relu`` kwarg) rather than inferred from
    ``has_bias``. The two are independent: the ``_bn_activation`` twins are
    the only apps with bias support, and they unconditionally apply ReLU, so
    ``has_bias=True, apply_relu=False`` has no matching app and is refused.
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


def from_torch_conv2d(
    layer: "torch.nn.Conv2d", *, apply_relu: bool = False,
) -> Conv2dDescription:
    """Build a :class:`Conv2dDescription` from a real ``torch.nn.Conv2d``.

    ``apply_relu`` must be passed explicitly by the caller -- see the
    ``apply_relu`` field's docstring for why it can't be inferred from the
    layer itself (a plain ``Conv2d`` has no activation concept).
    """
    def _one(v) -> int:
        if isinstance(v, tuple):
            if v[0] != v[1]:
                raise UnsupportedLayer(
                    f"non-square parameter {v} is not supported (kernel_size, "
                    "stride, padding, and dilation must be scalars or square "
                    "tuples)"
                )
            return v[0]
        return int(v)

    return Conv2dDescription(
        in_channels=layer.in_channels,
        out_channels=layer.out_channels,
        kernel_size=_one(layer.kernel_size),
        stride=_one(layer.stride),
        padding=_one(layer.padding),
        dilation=_one(layer.dilation),
        groups=layer.groups,
        has_bias=layer.bias is not None,
        apply_relu=apply_relu,
    )


def _validate(desc: Conv2dDescription) -> None:
    if desc.kernel_size != 3:
        raise UnsupportedLayer(
            f"kernel_size={desc.kernel_size} is not supported; only 3x3 "
            "kernels are implemented"
        )
    if desc.dilation != 1:
        raise UnsupportedLayer(
            f"dilation={desc.dilation} is not supported; only dilation=1 "
            "is implemented"
        )
    if desc.padding != 1:
        raise UnsupportedLayer(
            f"padding={desc.padding} is not supported; only padding=1 "
            "(\"same\" padding for a 3x3 kernel) is implemented"
        )
    if desc.stride not in (1, 2):
        raise UnsupportedLayer(
            f"stride={desc.stride} is not supported; only stride in {{1, 2}} "
            "is implemented"
        )
    if desc.groups not in (1, desc.in_channels):
        raise UnsupportedLayer(
            f"groups={desc.groups} is not supported; only groups=1 (plain "
            f"conv) or groups=in_channels={desc.in_channels} (fully "
            "depthwise) is implemented -- partial grouping is not"
        )
    if desc.groups == desc.in_channels and desc.in_channels != desc.out_channels:
        raise UnsupportedLayer(
            f"depthwise conv (groups=in_channels={desc.in_channels}) requires "
            f"out_channels == in_channels, got out_channels={desc.out_channels} "
            "(a depthwise multiplier > 1 is not implemented)"
        )
    if desc.stride == 2 and desc.groups == 1:
        raise UnsupportedLayer(
            "stride=2 for a plain (non-depthwise) conv is not implemented; "
            "only depthwise stride=2 (depthwise_conv_stride2_128) exists today"
        )
    if desc.has_bias and not desc.apply_relu:
        raise UnsupportedLayer(
            "has_bias=True with apply_relu=False has no matching app: the "
            "only apps with bias support (conv_universal_bn_activation, "
            "depthwise_conv_universal_bn_activation) unconditionally apply "
            "ReLU. Either pass apply_relu=True (if that activation is "
            "actually wanted) or drop the bias."
        )
    if desc.apply_relu and desc.stride == 2:
        raise UnsupportedLayer(
            "apply_relu=True is not supported for stride=2: "
            "depthwise_conv_stride2_128 has no bias/activation twin"
        )


# ============================================================================
# Dispatch + run
# ============================================================================

def run_layer(
    layer: "torch.nn.Conv2d",
    input_tensor: "torch.Tensor",
    *,
    apply_relu: bool = False,
    max_cycles: int = 5_000_000,
) -> "torch.Tensor":
    """Run ``layer`` on ``input_tensor`` through the matching IPU app.

    ``input_tensor`` must be ``[in_channels, height, width]`` (no batch
    dimension -- these apps process one image at a time) and already
    INT8-valued (see :func:`quantize_int8`). Returns an
    ``[out_channels, out_height, out_width]`` int8 tensor of the SAME true
    (unpadded) shape ``F.conv2d`` with ``padding=1`` would produce.

    ``apply_relu`` must be passed explicitly -- a plain ``torch.nn.Conv2d``
    has no activation concept, so it can never be inferred from ``layer``
    (see :class:`Conv2dDescription`'s docstring). If ``layer.bias`` is set,
    ``apply_relu=True`` is required (the only apps with bias support
    unconditionally apply ReLU).
    """
    import torch

    desc = from_torch_conv2d(layer, apply_relu=apply_relu)
    _validate(desc)

    in_ch, height, width = input_tensor.shape
    if in_ch != desc.in_channels:
        raise UnsupportedLayer(
            f"input has {in_ch} channels, layer expects {desc.in_channels}"
        )
    if height > 128 or width > 128:
        raise UnsupportedLayer(
            f"height={height}, width={width}: only spatial dims up to 128 "
            "are supported (via pad+truncate); conv_universal_wide384 "
            "handles wider images but is not wired into this adapter"
        )
    if desc.stride == 2 and height % 2 != 0:
        raise UnsupportedLayer(
            f"depthwise_conv_stride2_128 requires an even row count, got "
            f"height={height}"
        )

    input_np = quantize_int8(input_tensor)
    weight_np = quantize_int8(layer.weight)
    bias_np = quantize_int8(layer.bias) if desc.has_bias else None

    is_depthwise = desc.groups == desc.in_channels
    cols = _next_valid_cols(width)
    padded = pad_width_to_cols(input_np, cols)

    with tempfile.TemporaryDirectory(prefix="run_layer_") as tmp_s:
        tmp = Path(tmp_s)
        input_file = tmp / "input.bin"
        input_file.write_bytes(pack_input_chunked(padded))
        output_file = tmp / "output.bin"

        if desc.stride == 2:
            # is_depthwise is guaranteed True here (_validate refuses
            # stride=2 for a plain conv), and depthwise_conv_stride2_128 is
            # hardwired to cols=128 -- pad all the way there regardless of
            # what _next_valid_cols picked for a possibly-smaller width.
            if cols != 128:
                padded = pad_width_to_cols(input_np, 128)
                input_file.write_bytes(pack_input_chunked(padded))
            kernel_file = tmp / "kernel.bin"
            # depthwise_conv_stride2_128 -> depthwise_conv_universal's own
            # kernel_path format: channels*9 raw INT8 bytes, no bias (stride2
            # has no BN/bias twin yet; _validate already refused apply_relu=True
            # for stride=2, and has_bias without apply_relu, above).
            kernel_file.write_bytes(weight_np.reshape(desc.out_channels, 9).tobytes())
            app = DepthwiseConvStride2_128App(
                input_path=input_file, kernel_path=kernel_file,
                output_path=output_file, rows=height, channels=desc.out_channels,
            )
            app.run(max_cycles=max_cycles)
            true_out_rows = height // 2
            true_out_cols = (width + 1) // 2
            out_arr = unpack_output_chunked(
                output_file.read_bytes(), desc.out_channels,
                math.ceil(true_out_rows / 2) * 2, 64,
            )
            out_arr = out_arr[:, :true_out_rows, :true_out_cols]
            return torch.from_numpy(out_arr.copy())

        # stride == 1: conv_universal / depthwise_conv_universal (+BN twins)
        bin_path = tmp / "assembled.bin"
        if is_depthwise:
            kernel_file = tmp / "kernel.bin"
            kernel_file.write_bytes(weight_np.reshape(desc.out_channels, 9).tobytes())
            if desc.apply_relu:
                assemble_to_bin_file(_DW_BN_ASM_PATH.read_text(), str(bin_path))
                app = DepthwiseConvUniversalBnActivationApp(
                    inst_path=bin_path, input_path=input_file,
                    kernel_path=kernel_file, bias=bias_np, output_path=output_file,
                    dtype="INT8", rows=height, cols=cols, channels=desc.out_channels,
                )
                out_chunk_bytes = _DW_BN_OUTPUT_CHUNK_BYTES
            else:
                assemble_to_bin_file(_DW_ASM_PATH.read_text(), str(bin_path))
                app = DepthwiseConvUniversalApp(
                    inst_path=bin_path, input_path=input_file,
                    kernel_path=kernel_file, output_path=output_file,
                    dtype="INT8", rows=height, cols=cols, channels=desc.out_channels,
                )
                out_chunk_bytes = _DW_OUTPUT_CHUNK_BYTES
        else:
            if desc.apply_relu:
                assemble_to_bin_file(_CONV_BN_ASM_PATH.read_text(), str(bin_path))
                app = ConvUniversalBnActivationApp(
                    inst_path=bin_path, input_path=input_file, kernel=weight_np,
                    bias=bias_np, output_path=output_file, dtype="INT8",
                    rows=height, cols=cols, in_channels=desc.in_channels,
                    out_channels=desc.out_channels,
                )
                out_chunk_bytes = _CONV_BN_OUTPUT_CHUNK_BYTES
            else:
                assemble_to_bin_file(_CONV_ASM_PATH.read_text(), str(bin_path))
                app = ConvUniversalApp(
                    inst_path=bin_path, input_path=input_file, kernel=weight_np,
                    output_path=output_file, dtype="INT8",
                    rows=height, cols=cols, in_channels=desc.in_channels,
                    out_channels=desc.out_channels,
                )
                out_chunk_bytes = _CONV_OUTPUT_CHUNK_BYTES

        app.run(max_cycles=max_cycles)
        assert out_chunk_bytes == CHUNK_BYTES  # all stride-1 apps use 128B chunks
        out_arr = unpack_output_chunked(
            output_file.read_bytes(), desc.out_channels, height, cols,
        )
        out_arr = truncate_width(out_arr, width)
        return torch.from_numpy(out_arr.copy())


def resolve(desc: Conv2dDescription) -> str:
    """Framework-free check: return a label for the app ``desc`` would route
    to, or raise :class:`UnsupportedLayer` with the specific reason.

    Does not run anything -- use :func:`run_layer` for that.
    """
    _validate(desc)
    is_depthwise = desc.groups == desc.in_channels
    if desc.stride == 2:
        return "depthwise_conv_stride2_128"
    if is_depthwise:
        return (
            "depthwise_conv_universal_bn_activation" if desc.apply_relu
            else "depthwise_conv_universal"
        )
    return (
        "conv_universal_bn_activation" if desc.apply_relu else "conv_universal"
    )
