"""Element values as a format holds them: its range, rounding into it, its bytes.

One place for the harness (the manifest's ``range`` check, case inputs,
packing): every per-format branch is here, none in the callers. The
element codec itself moves to ``ipu_isa.formats`` in E04-13; until then FP8
goes through the emulator's public encode and decode, which round to nearest
and saturate.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_emu.ipu_math import DType, fp8_bytes_to_fp32, fp32_to_fp8_bytes
from ipu_isa.formats import FORMATS, ElementFormat

Array = npt.NDArray[Any]

_FP32_MAX = float(np.finfo(np.float32).max)


@functools.cache
def limits(fmt: ElementFormat) -> tuple[float, float]:
    """The lowest and highest finite value ``fmt`` holds.

    Args:
        fmt: The element format.

    Returns:
        ``(lowest, highest)``.
    """
    info = FORMATS[fmt]
    if not info.is_float:
        half = 1 << (info.element_bits - 1)
        return float(-half), float(half - 1)
    if fmt is ElementFormat.FP32:
        return -_FP32_MAX, _FP32_MAX
    decoded = fp8_bytes_to_fp32(bytes(range(256)), DType(info.exp_bits))
    top = float(np.max(decoded[np.isfinite(decoded)]))
    return -top, top


def held(values: Array, fmt: ElementFormat) -> Array:
    """``values`` as ``fmt`` holds them: rounded to nearest, saturated to its range.

    Args:
        values: Finite numbers (the caller checks; NaN has no defined result).
        fmt: The element format.

    Returns:
        An int8 or int32 array for an integer format, else float32.
    """
    info = FORMATS[fmt]
    lo, hi = limits(fmt)
    clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi)
    if not info.is_float:
        return np.rint(clipped).astype(np.int8 if info.element_bits == 8 else np.int32)
    as_fp32 = clipped.astype(np.float32)
    if fmt is ElementFormat.FP32:
        return as_fp32
    dtype = DType(info.exp_bits)
    flat: Array = fp8_bytes_to_fp32(fp32_to_fp8_bytes(as_fp32.ravel(), dtype), dtype)
    return np.reshape(flat, as_fp32.shape)


def _numpy_dtype(fmt: ElementFormat) -> str | None:
    """The little-endian NumPy type of a format NumPy holds natively, else None (FP8)."""
    info = FORMATS[fmt]
    if fmt is ElementFormat.FP32:
        return "<f4"
    if not info.is_float:
        return {8: "<i1", 32: "<i4"}[info.element_bits]
    return None


def encode(values: Array, fmt: ElementFormat) -> bytes:
    """``values`` as ``fmt``'s bytes, element after element.

    Args:
        values: Finite numbers; rounded to the format first (:func:`held`).
        fmt: The element format.

    Returns:
        ``values.size * element_bytes(fmt)`` bytes, little-endian.
    """
    in_format = held(values, fmt).ravel()
    native = _numpy_dtype(fmt)
    if native is not None:
        return in_format.astype(native).tobytes()
    return fp32_to_fp8_bytes(in_format.astype(np.float32), DType(FORMATS[fmt].exp_bits))


def decode(raw: bytes, fmt: ElementFormat) -> Array:
    """``fmt``'s bytes as float64 values, element after element (inverse of :func:`encode`).

    Args:
        raw: A whole number of ``fmt`` elements.
        fmt: The element format.

    Returns:
        A flat float64 array.
    """
    native = _numpy_dtype(fmt)
    if native is not None:
        return np.frombuffer(raw, dtype=native).astype(np.float64)
    decoded: Array = fp8_bytes_to_fp32(raw, DType(FORMATS[fmt].exp_bits))
    return decoded.astype(np.float64)
