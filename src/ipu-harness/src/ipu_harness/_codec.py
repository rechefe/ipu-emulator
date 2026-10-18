"""Element values as a format holds them: its range, rounding into it, its bytes.

One place for the harness (the manifest's ``range`` check, case inputs,
packing): every per-format branch is here, none in the callers. FP8 goes
through the element codec in ``ipu_isa.formats`` (E04-13), which rounds to
nearest and saturates.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import numpy.typing as npt

from ipu_isa.formats import FORMATS, ElementFormat, element_values, encode_element

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
    top = float(codes(fmt)[-1])
    return -top, top


@functools.cache
def codes(fmt: ElementFormat) -> Array:
    """Every finite value of an 8-bit float format, ascending: a value's index
    is its place in code order (for distances in codes).

    Args:
        fmt: An 8-bit float format.

    Returns:
        The values, as a float64 array without duplicates (+0 and -0 are one).
    """
    decoded = np.array(element_values(fmt), dtype=np.float32)
    values: Array = np.unique(decoded[np.isfinite(decoded)].astype(np.float64))
    values.flags.writeable = False  # cached: shared by every caller
    return values


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
    flat: Array = _fp8_decode(_fp8_encode(as_fp32.ravel(), fmt), fmt)
    return np.reshape(flat, as_fp32.shape)


def _fp8_encode(values: Array, fmt: ElementFormat) -> bytes:
    """float32 ``values`` as 8-bit float codes, one element at a time."""
    return bytes(encode_element(float(v), fmt) for v in values.astype(np.float32))


def _fp8_decode(raw: bytes, fmt: ElementFormat) -> Array:
    """8-bit float codes as float32 values (a table lookup)."""
    table = np.array(element_values(fmt), dtype=np.float32)
    decoded: Array = table[np.frombuffer(raw, dtype=np.uint8)]
    return decoded


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
    native = _numpy_dtype(fmt)
    if native is not None:
        return held(values, fmt).ravel().astype(native).tobytes()
    # FP8: clipped as held() clips, so the bytes are held()'s values; the encoder rounds.
    lo, hi = limits(fmt)
    clipped = np.clip(np.asarray(values, dtype=np.float64), lo, hi).ravel()
    return _fp8_encode(clipped.astype(np.float32), fmt)


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
    return _fp8_decode(raw, fmt).astype(np.float64)
