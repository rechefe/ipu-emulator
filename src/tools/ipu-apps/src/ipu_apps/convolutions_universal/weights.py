"""Weight packers for the universal convolution apps.

Packs PyTorch-style numpy weight tensors into the dense XMEM byte layouts
consumed by the IPU's 3x3 and 1x1 convolution apps.

Layouts
-------
Dense 3x3 (conv + depthwise):
    One 128-byte block holds ``FPB = 128 // (K*K)`` filter slots. For K=3,
    FPB = 14 (14 * 9 = 126 data bytes, 2 bytes padding). Per filter, blocks
    are laid out consecutively; the last block zero-pads any unused slots.

Pointwise universal / pointwise 8x8:
    Exactly reproduce the byte layouts previously built by the app harnesses'
    ``_pack_kernel`` / ``_build_kernel_data`` methods. This module is the
    forward-looking single place those layouts live.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from ipu_emu.ipu_math import DType, fp32_to_fp8_bytes


def _fpb(kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    k2 = kernel_size * kernel_size
    if k2 > 128:
        raise ValueError(f"kernel_size*kernel_size must be <= 128, got {k2}")
    return 128 // k2


def _ensure_ndim(arr: np.ndarray, expected: Union[int, tuple[int, ...]], name: str) -> None:
    expected_tuple = (expected,) if isinstance(expected, int) else expected
    if arr.ndim not in expected_tuple:
        raise ValueError(
            f"{name} must have ndim in {expected_tuple}, got shape {arr.shape}"
        )


def _validate_and_cast_to_bytes(arr: np.ndarray, dtype: DType) -> bytes:
    """Convert a validated numpy array into raw IPU wire bytes for ``dtype``.

    Contract:
      - ``DType.INT8`` requires ``arr.dtype == np.int8``.
      - FP8 formats (``DType.E1``..``DType.E7``) require ``arr.dtype == np.float32``.
    """
    contiguous = np.ascontiguousarray(arr)
    if dtype == DType.INT8:
        if contiguous.dtype != np.int8:
            raise ValueError(
                f"DType.INT8 requires numpy dtype int8, got {contiguous.dtype}"
            )
        return contiguous.view(np.uint8).tobytes()
    # FP8 path
    if contiguous.dtype != np.float32:
        raise ValueError(
            f"{dtype!r} requires numpy dtype float32, got {contiguous.dtype}"
        )
    return fp32_to_fp8_bytes(contiguous.ravel(order="C"), dtype)


def _squeeze_pointwise(weights: np.ndarray) -> np.ndarray:
    """Accept [out, in] or [out, in, 1, 1]; return [out, in]."""
    if weights.ndim == 2:
        return weights
    if weights.ndim == 4:
        if weights.shape[2] != 1 or weights.shape[3] != 1:
            raise ValueError(
                f"pointwise weights must have K=1 trailing dims, got shape {weights.shape}"
            )
        return weights.reshape(weights.shape[0], weights.shape[1])
    raise ValueError(
        f"pointwise weights must have ndim 2 or 4, got shape {weights.shape}"
    )


def pack_conv_weights_dense(
    weights: np.ndarray,
    dtype: DType,
    *,
    kernel_size: int = 3,
) -> bytes:
    """Pack ``[out_ch, in_ch, K, K]`` weights into dense 128-byte blocks.

    One 128-byte block holds ``FPB = 128 // (K*K)`` input-channel slots of one
    output filter. Taps are stored row-major within each slot (``tap = dr*K + dc``).
    Blocks for filter 0 come first (``ceil(in_ch / FPB)`` of them), then filter 1,
    and so on. The last block of each filter zero-pads any unused slots.
    """
    _ensure_ndim(weights, 4, "weights")
    out_ch, in_ch, kh, kw = weights.shape
    if kh != kernel_size or kw != kernel_size:
        raise ValueError(
            f"weights trailing dims must be ({kernel_size}, {kernel_size}), "
            f"got ({kh}, {kw})"
        )

    fpb = _fpb(kernel_size)
    k2 = kernel_size * kernel_size
    raw = _validate_and_cast_to_bytes(weights, dtype)
    # raw indexing: raw[((f*in_ch + ic)*K + dr)*K + dc] == raw[(f*in_ch + ic)*k2 + tap]

    blocks_per_filter = math.ceil(in_ch / fpb)
    total = out_ch * blocks_per_filter * 128
    packed = bytearray(total)

    for f in range(out_ch):
        for b in range(blocks_per_filter):
            block_base = (f * blocks_per_filter + b) * 128
            for s in range(fpb):
                ic = b * fpb + s
                if ic >= in_ch:
                    break
                src = (f * in_ch + ic) * k2
                dst = block_base + s * k2
                packed[dst:dst + k2] = raw[src:src + k2]

    return bytes(packed)


def pack_depthwise_weights_dense(
    weights: np.ndarray,
    dtype: DType,
    *,
    kernel_size: int = 3,
) -> bytes:
    """Pack ``[channels, 1, K, K]`` depthwise weights into dense 128-byte blocks.

    Each block holds ``FPB = 128 // (K*K)`` independent channels. Taps are
    row-major within each slot. The last block zero-pads any unused slots.
    """
    _ensure_ndim(weights, 4, "weights")
    channels, mid, kh, kw = weights.shape
    if mid != 1:
        raise ValueError(
            f"depthwise weights must have shape [channels, 1, K, K], got {weights.shape}"
        )
    if kh != kernel_size or kw != kernel_size:
        raise ValueError(
            f"weights trailing dims must be ({kernel_size}, {kernel_size}), "
            f"got ({kh}, {kw})"
        )

    fpb = _fpb(kernel_size)
    k2 = kernel_size * kernel_size
    raw = _validate_and_cast_to_bytes(weights, dtype)
    # raw indexing: raw[c*k2 + tap]

    num_blocks = math.ceil(channels / fpb)
    total = num_blocks * 128
    packed = bytearray(total)

    for b in range(num_blocks):
        block_base = b * 128
        for s in range(fpb):
            c = b * fpb + s
            if c >= channels:
                break
            src = c * k2
            dst = block_base + s * k2
            packed[dst:dst + k2] = raw[src:src + k2]

    return bytes(packed)


def _compute_G_pointwise(in_channels: int) -> int:
    """Largest power-of-2 dividing in_channels, capped at 128."""
    g = in_channels & (-in_channels)
    return min(g, 128)


def pack_pointwise_weights_universal(
    weights: np.ndarray,
    in_channels: int,
    out_channels: int,
    dtype: DType,
) -> bytes:
    """Pack 1x1 conv weights for the universal pointwise app.

    Accepts ``[out_ch, in_ch]`` or ``[out_ch, in_ch, 1, 1]``. Reproduces the
    byte layout produced by ``PointwiseConvUniversalApp._pack_kernel``.
    """
    w = _squeeze_pointwise(weights)
    if w.shape != (out_channels, in_channels):
        raise ValueError(
            f"weights shape {w.shape} does not match "
            f"(out_channels={out_channels}, in_channels={in_channels})"
        )

    G = _compute_G_pointwise(in_channels)
    if G < 4:
        raise ValueError(
            f"in_channels ({in_channels}) must be divisible by at least 4"
        )
    num_groups = in_channels // G
    oc_per_reg = 128 // G
    if out_channels % (2 * oc_per_reg) != 0:
        raise ValueError(
            f"out_channels ({out_channels}) must be divisible by "
            f"{2 * oc_per_reg} (2 * 128/{G})"
        )
    num_batches = out_channels // (2 * oc_per_reg)

    raw = _validate_and_cast_to_bytes(w, dtype)
    # raw indexing: raw[oc * in_channels + ic]

    total_size = num_batches * 2 * num_groups * 128
    packed = bytearray(total_size)

    for batch in range(num_batches):
        for half in range(2):
            for group in range(num_groups):
                dest_base = ((batch * 2 + half) * num_groups + group) * 128
                for j in range(oc_per_reg):
                    oc = batch * 2 * oc_per_reg + half * oc_per_reg + j
                    for i in range(G):
                        ic = group * G + i
                        packed[dest_base + j * G + i] = raw[oc * in_channels + ic]

    return bytes(packed)


def pack_pointwise_weights_8x8(
    weights: np.ndarray,
    dtype: DType,
) -> bytes:
    """Pack 1x1 conv weights for the paired-output 8x8 pointwise app.

    Accepts ``[out_ch, in_ch]`` or ``[out_ch, in_ch, 1, 1]``. Reproduces the
    byte layout produced by ``pointwise_8x8._build_kernel_data``.
    """
    w = _squeeze_pointwise(weights)
    out_channels, in_channels = w.shape
    if in_channels < 2 or in_channels % 2 != 0:
        raise ValueError(f"in_channels must be even and >= 2, got {in_channels}")
    if out_channels < 2 or out_channels % 2 != 0:
        raise ValueError(f"out_channels must be even and >= 2, got {out_channels}")

    raw = _validate_and_cast_to_bytes(w, dtype)
    # raw indexing: raw[oc * in_channels + ic]

    ic_pairs = in_channels // 2
    oc_pairs = out_channels // 2
    blocks_per_pair = math.ceil(ic_pairs / 32)
    kernel_bytes_per_pair = blocks_per_pair * 128

    packed = bytearray(oc_pairs * kernel_bytes_per_pair)

    for p in range(oc_pairs):
        f0 = 2 * p
        f1 = 2 * p + 1
        for j in range(ic_pairs):
            block = j // 32
            pos_in_block = j % 32
            dst = p * kernel_bytes_per_pair + block * 128 + pos_in_block * 4
            ic_even = 2 * j
            ic_odd = 2 * j + 1
            packed[dst] = raw[f0 * in_channels + ic_even]
            packed[dst + 1] = raw[f0 * in_channels + ic_odd]
            packed[dst + 2] = raw[f1 * in_channels + ic_even]
            packed[dst + 3] = raw[f1 * in_channels + ic_odd]

    return bytes(packed)
