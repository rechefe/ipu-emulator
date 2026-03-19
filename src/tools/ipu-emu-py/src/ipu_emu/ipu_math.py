"""IPU math operations — Python port of ipu_math.c.

Uses ``ml_dtypes`` for well-known FP8 conversions (E4M3fn, E5M2, E3M4).
For all other (exp_bits, man_bits) combinations (0–7 each), a generic
IEEE-style FP8 converter is used, enabling fully configurable data types.

Dtype encoding
--------------
* ``DType.INT8 = 0``  — 8-bit signed integer arithmetic.
* FP8 types are encoded as ``(exp_bits << 3) | man_bits`` (always > 0
  for valid FP8 formats).  Use :func:`make_fp8_dtype` to construct these
  values.

Examples::

    dtype_e4m3 = make_fp8_dtype(4, 3)  # == DType.FP8_E4M3
    dtype_e5m2 = make_fp8_dtype(5, 2)  # == DType.FP8_E5M2
    dtype_e3m4 = make_fp8_dtype(3, 4)  # custom — 3 exp, 4 mantissa bits
    dtype_e2m5 = make_fp8_dtype(2, 5)  # custom — 2 exp, 5 mantissa bits
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np
from ml_dtypes import float8_e3m4, float8_e4m3fn, float8_e5m2


# ---------------------------------------------------------------------------
# FP8 dtype encoding helpers
# ---------------------------------------------------------------------------


def make_fp8_dtype(exp_bits: int, man_bits: int) -> int:
    """Return the integer dtype code for an FP8 format with given field widths.

    The encoding is ``(exp_bits << 3) | man_bits``, which is always > 0 for
    any valid (exp, man) pair where at least one field is non-zero.

    Args:
        exp_bits: Number of exponent bits (0–7).
        man_bits: Number of mantissa bits (0–7).

    Returns:
        Non-zero integer dtype code for use with :func:`ipu_mult`,
        :func:`fp32_to_fp8_bytes`, etc.

    Raises:
        ValueError: If either field is out of range [0, 7] or both are zero.
    """
    if not (0 <= exp_bits <= 7):
        raise ValueError(f"exp_bits must be 0–7, got {exp_bits}")
    if not (0 <= man_bits <= 7):
        raise ValueError(f"man_bits must be 0–7, got {man_bits}")
    if exp_bits == 0 and man_bits == 0:
        raise ValueError("exp_bits and man_bits cannot both be zero")
    return (exp_bits << 3) | man_bits


def get_fp8_exp_bits(dtype: int) -> int:
    """Extract the exponent-bit count from a FP8 dtype code."""
    return (dtype >> 3) & 0x7


def get_fp8_man_bits(dtype: int) -> int:
    """Extract the mantissa-bit count from a FP8 dtype code."""
    return dtype & 0x7


def is_fp8_dtype(dtype: int) -> bool:
    """Return ``True`` if *dtype* encodes an FP8 format (i.e. not INT8)."""
    return dtype != 0


# ---------------------------------------------------------------------------
# DType enum
# ---------------------------------------------------------------------------


class DType(IntEnum):
    """Data type enumeration — matches ``ipu_math__dtype_t`` in C.

    INT8 is stored as 0.  FP8 types are stored as ``(exp_bits << 3) | man_bits``
    (see :func:`make_fp8_dtype`).
    """
    INT8 = 0
    FP8_E4M3 = (4 << 3) | 3   # 35
    FP8_E5M2 = (5 << 3) | 2   # 42


# Lookup from dtype code to the ml_dtypes numpy dtype for well-known formats.
# These use ml_dtypes for maximum hardware accuracy; all other FP8 combinations
# fall back to the generic converter.
_FP8_NUMPY_DTYPE: dict[int, object] = {
    DType.FP8_E4M3: float8_e4m3fn,
    DType.FP8_E5M2: float8_e5m2,
    make_fp8_dtype(3, 4): float8_e3m4,
}


# ---------------------------------------------------------------------------
# Generic FP8 ↔ FP32 converters (IEEE-style, configurable exp/man widths)
# ---------------------------------------------------------------------------


def fp8_to_fp32_generic(raw: int, exp_bits: int, man_bits: int) -> float:
    """Decode a raw FP8 byte to a Python ``float`` for arbitrary field widths.

    Follows IEEE conventions:

    * ``exp_field == 0`` → zero (``man_field == 0``) or subnormal.
    * ``exp_field == (2**exp_bits) - 1`` → infinity (``man_field == 0``) or NaN.
    * Otherwise → normal number ``(-1)^s * 2^(exp_field-bias) * (1 + man_field/2^man_bits)``.

    When ``exp_bits == 0`` the value is treated as a signed fixed-point integer
    ``(-1)^s * man_field``.

    Args:
        raw:      Raw 8-bit integer (only the low ``1 + exp_bits + man_bits``
                  bits are used).
        exp_bits: Number of exponent bits (0–7).
        man_bits: Number of mantissa bits (0–7).
    """
    if exp_bits == 0:
        # No exponent field: signed integer from the mantissa bits.
        sign = (raw >> man_bits) & 1
        mant_field = raw & ((1 << man_bits) - 1) if man_bits > 0 else 0
        value = float(mant_field)
        return -value if sign else value

    bias = (1 << (exp_bits - 1)) - 1
    sign_shift = exp_bits + man_bits
    sign = (raw >> sign_shift) & 1
    exp_field = (raw >> man_bits) & ((1 << exp_bits) - 1)
    mant_field = raw & ((1 << man_bits) - 1) if man_bits > 0 else 0

    max_exp_field = (1 << exp_bits) - 1

    if exp_field == max_exp_field:
        # All exponent bits set → infinity or NaN.
        if mant_field == 0:
            return math.copysign(math.inf, -1.0 if sign else 1.0)
        return math.nan

    if exp_field == 0:
        # Zero or subnormal: ``0.mant_field * 2^(1 - bias)``.
        if mant_field == 0:
            return -0.0 if sign else 0.0
        value = float(mant_field) / (1 << man_bits) * (2.0 ** (1 - bias))
    else:
        # Normal: ``(1 + mant_field / 2^man_bits) * 2^(exp_field - bias)``.
        value = (1.0 + float(mant_field) / (1 << man_bits)) * (2.0 ** (exp_field - bias))

    return -value if sign else value


def _fp8_max_normal(exp_bits: int, man_bits: int, max_exp_field: int) -> float:
    """Return the largest finite value representable in FP8(exp_bits, man_bits)."""
    if man_bits == 0:
        return 2.0 ** max_exp_field
    max_mant = 1.0 + ((1 << man_bits) - 1) / (1 << man_bits)
    return max_mant * (2.0 ** max_exp_field)


def fp32_to_fp8_generic(value: float, exp_bits: int, man_bits: int) -> int:
    """Encode a Python ``float`` as a raw FP8 byte for arbitrary field widths.

    Follows IEEE conventions with round-to-nearest-even on the mantissa and
    saturation (rather than overflow to infinity) when the value exceeds the
    representable range.

    When ``exp_bits == 0`` the value is quantized to the nearest non-negative
    integer in ``[0, 2^man_bits - 1]`` and stored as a signed fixed-point byte.

    Args:
        value:    Python float to encode.
        exp_bits: Number of exponent bits (0–7).
        man_bits: Number of mantissa bits (0–7).

    Returns:
        Raw byte (integer in ``[0, 255]``).
    """
    if exp_bits == 0:
        sign = 1 if value < 0 else 0
        x = int(round(abs(value)))
        max_mant = (1 << man_bits) - 1 if man_bits > 0 else 0
        x = min(x, max_mant)
        return (sign << man_bits) | x

    bias = (1 << (exp_bits - 1)) - 1
    max_exp_field = (1 << exp_bits) - 2  # reserve all-ones for inf/NaN

    sign = 1 if value < 0 else 0
    x = abs(value)

    if math.isnan(value):
        # NaN → all bits set.
        total_bits = 1 + exp_bits + man_bits
        return (1 << total_bits) - 1

    if x == 0.0:
        return sign << (exp_bits + man_bits)

    if math.isinf(x) or x > _fp8_max_normal(exp_bits, man_bits, max_exp_field):
        # Overflow → saturate to largest finite value.
        raw = (sign << (exp_bits + man_bits)) | (max_exp_field << man_bits)
        raw |= (1 << man_bits) - 1 if man_bits > 0 else 0
        return raw

    exp = math.floor(math.log2(x))
    exp_field = exp + bias

    if exp_field <= 0:
        # Subnormal range: encode as ``mant_field / 2^man_bits * 2^(1 - bias)``.
        if man_bits == 0:
            return sign << (exp_bits + man_bits)
        scaled = x * (2.0 ** (man_bits - 1 + bias))
        mant_field = round(scaled)  # round-to-nearest-even
        if mant_field >= (1 << man_bits):
            # Round-up promotes to smallest normal.
            return (sign << (exp_bits + man_bits)) | (1 << man_bits)
        return (sign << (exp_bits + man_bits)) | max(mant_field, 0)

    # Normal number.
    if man_bits == 0:
        mant_field = 0
    else:
        frac = x / (2.0 ** exp) - 1.0  # in [0, 1)
        scaled = frac * (1 << man_bits)
        mant_field = round(scaled)  # round-to-nearest-even
        if mant_field >= (1 << man_bits):
            # Round-up overflows mantissa → carry into exponent.
            mant_field = 0
            exp_field += 1
            if exp_field > max_exp_field:
                # Saturate to max finite.
                return (
                    (sign << (exp_bits + man_bits))
                    | (max_exp_field << man_bits)
                    | ((1 << man_bits) - 1)
                )

    return (sign << (exp_bits + man_bits)) | (exp_field << man_bits) | mant_field


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _int8_to_signed(val: int) -> int:
    """Interpret an unsigned byte as a signed int8."""
    return val if val < 128 else val - 256


def _fp8_to_float(val: int, dtype: int) -> float:
    """Convert a raw FP8 byte to a Python float.

    Uses ``ml_dtypes`` for well-known formats; falls back to the generic
    converter for custom (exp_bits, man_bits) combinations.
    """
    np_dtype = _FP8_NUMPY_DTYPE.get(dtype)
    if np_dtype is not None:
        raw = np.array(val, dtype=np.uint8)
        return float(raw.view(np_dtype))
    exp_bits = get_fp8_exp_bits(dtype)
    man_bits = get_fp8_man_bits(dtype)
    return fp8_to_fp32_generic(val, exp_bits, man_bits)


# ---------------------------------------------------------------------------
# Public conversion API
# ---------------------------------------------------------------------------


def fp32_to_fp8_bytes(fp32_values: np.ndarray, dtype: int) -> bytes:
    """Convert an array of FP32 values to FP8 bytes.

    Uses ``ml_dtypes`` for well-known formats (E4M3fn, E5M2, E3M4) and the
    generic converter for custom FP8 formats.  Raises :exc:`ValueError` for
    ``DType.INT8``.

    Args:
        fp32_values: 1-D (or flat) numpy array of ``float32`` values.
        dtype:       FP8 dtype code (see :func:`make_fp8_dtype`).

    Raises:
        ValueError: If *dtype* is ``DType.INT8`` (0).
    """
    if dtype == DType.INT8:
        raise ValueError(
            f"fp32_to_fp8_bytes only supports FP8 dtypes, got {dtype!r}"
        )
    np_dtype = _FP8_NUMPY_DTYPE.get(dtype)
    if np_dtype is not None:
        fp8_arr = fp32_values.astype(np.float32).astype(np_dtype)
        return fp8_arr.view(np.uint8).tobytes()

    # Generic path for custom FP8 formats.
    exp_bits = get_fp8_exp_bits(dtype)
    man_bits = get_fp8_man_bits(dtype)
    flat = fp32_values.astype(np.float32).ravel()
    out = bytearray(len(flat))
    for i, v in enumerate(flat):
        out[i] = fp32_to_fp8_generic(float(v), exp_bits, man_bits)
    return bytes(out)


def fp8_bytes_to_fp32(raw: bytes, dtype: int) -> np.ndarray:
    """Convert raw FP8 bytes back to an FP32 numpy array.

    Uses ``ml_dtypes`` for well-known formats and the generic converter for
    custom FP8 formats.  Raises :exc:`ValueError` for ``DType.INT8``.

    Args:
        raw:   Raw bytes, one byte per FP8 value.
        dtype: FP8 dtype code (see :func:`make_fp8_dtype`).

    Raises:
        ValueError: If *dtype* is ``DType.INT8`` (0).
    """
    if dtype == DType.INT8:
        raise ValueError(
            f"fp8_bytes_to_fp32 only supports FP8 dtypes, got {dtype!r}"
        )
    np_dtype = _FP8_NUMPY_DTYPE.get(dtype)
    if np_dtype is not None:
        arr = np.frombuffer(raw, dtype=np.uint8).view(np_dtype)
        return arr.astype(np.float32)

    # Generic path for custom FP8 formats.
    exp_bits = get_fp8_exp_bits(dtype)
    man_bits = get_fp8_man_bits(dtype)
    out = np.empty(len(raw), dtype=np.float32)
    for i, b in enumerate(raw):
        out[i] = fp8_to_fp32_generic(b, exp_bits, man_bits)
    return out


def ipu_mult(a_byte: int, b_byte: int, dtype: int) -> int | float:
    """Multiply two bytes according to *dtype*, returning int32 or float.

    Mirrors ``ipu_math__mult`` from C.
    """
    if dtype == DType.INT8:
        a = _int8_to_signed(a_byte)
        b = _int8_to_signed(b_byte)
        return a * b  # int32 result
    if is_fp8_dtype(dtype):
        return _fp8_to_float(a_byte, dtype) * _fp8_to_float(b_byte, dtype)
    raise ValueError(f"Unsupported dtype: {dtype}")


def ipu_add(a_word: int | float, b_word: int | float, dtype: int) -> int | float:
    """Add two accumulator-width values according to *dtype*.

    Mirrors ``ipu_math__add`` from C.
    """
    if dtype == DType.INT8:
        # Both are int32 — wrap to 32 bits with two's complement
        result = (a_word + b_word) & 0xFFFFFFFF
        return result - 0x100000000 if result >= 0x80000000 else result
    # FP8 variants: accumulator is float
    return a_word + b_word
