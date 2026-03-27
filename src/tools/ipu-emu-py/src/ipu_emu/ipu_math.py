"""IPU math operations — Python port of ipu_math.c.

Supports all e(x)m(8-x) 8-bit data types where x is the number of exponent
bits (0–7).  x=0 is the special integer mode (INT8); x=1..7 are FP8 formats
with IEEE-style encoding: 1 sign bit, x exponent bits, (7-x) mantissa bits,
and bias = 2^(x-1) - 1.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np


class DType(IntEnum):
    """8-bit data type.

    The integer value equals the number of exponent bits (x).
    INT8 (x=0) is the integer mode; E1..E7 are FP8 formats e(x)m(8-x)
    where "m" counts sign + mantissa bits, giving 1 sign + x exponent
    + (7-x) stored mantissa bits = 8 bits total.
    """
    INT8 = 0  # integer mode: 8-bit signed (no exponent)
    E1 = 1    # fp8_e1: 1 exp bit,  6 mantissa bits
    E2 = 2    # fp8_e2: 2 exp bits, 5 mantissa bits
    E3 = 3    # fp8_e3: 3 exp bits, 4 mantissa bits
    E4 = 4    # fp8_e4: 4 exp bits, 3 mantissa bits  (formerly FP8_E4M3)
    E5 = 5    # fp8_e5: 5 exp bits, 2 mantissa bits  (formerly FP8_E5M2)
    E6 = 6    # fp8_e6: 6 exp bits, 1 mantissa bit
    E7 = 7    # fp8_e7: 7 exp bits, 0 mantissa bits


def _int8_to_signed(val: int) -> int:
    """Interpret an unsigned byte as a signed int8."""
    return val if val < 128 else val - 256


def _fp8_max_finite(exp_bits: int, man_bits: int) -> int:
    """Return the unsigned byte encoding of the largest finite FP8 value (positive)."""
    max_exp_raw = (1 << exp_bits) - 1  # all-ones = NaN, so largest normal is max_exp_raw-1
    return ((max_exp_raw - 1) << man_bits) | ((1 << man_bits) - 1)


def _fp8_decode_fields(byte_val: int, exp_bits: int) -> tuple[int, int, int]:
    """Split an FP8 byte into (sign, exp_raw, man_raw)."""
    man_bits = 7 - exp_bits
    sign = (byte_val >> 7) & 1
    exp_raw = (byte_val >> man_bits) & ((1 << exp_bits) - 1)
    man_raw = byte_val & ((1 << man_bits) - 1)
    return sign, exp_raw, man_raw


def _fp8_magnitude(exp_raw: int, man_raw: int, exp_bits: int) -> float:
    """Decode the unsigned magnitude of a normal or subnormal FP8 value."""
    man_bits = 7 - exp_bits
    bias = (1 << (exp_bits - 1)) - 1
    if exp_raw == 0:
        # Subnormal: 0.man * 2^(1-bias)
        return (man_raw / (1 << man_bits)) * (2.0 ** (1 - bias))
    # Normal: 1.man * 2^(exp-bias)
    return (1 + man_raw / (1 << man_bits)) * (2.0 ** (exp_raw - bias))


def _fp8_to_float32_scalar(byte_val: int, exp_bits: int) -> float:
    """Decode one FP8 byte to a Python float.

    Format: 1 sign | exp_bits exponent | (7-exp_bits) mantissa.
    All-ones exponent encodes NaN; zero exponent encodes subnormals.
    """
    sign, exp_raw, man_raw = _fp8_decode_fields(byte_val, exp_bits)
    max_exp = (1 << exp_bits) - 1

    if exp_raw == max_exp:
        return float("nan")

    value = _fp8_magnitude(exp_raw, man_raw, exp_bits)
    return -value if sign else value


def _fp8_encode_subnormal(val: float, exp_bits: int, man_bits: int, sign: int) -> int:
    """Encode a subnormal FP8 value from a positive float magnitude."""
    bias = (1 << (exp_bits - 1)) - 1
    max_man = (1 << man_bits) - 1
    # val = (man_int / 2^man_bits) * 2^(1-bias)  →  man_int = val * 2^(man_bits+bias-1)
    man_int = round(val * (2.0 ** (man_bits + bias - 1)))
    man_int = max(0, min(man_int, max_man))
    return (sign << 7) | man_int


def _fp8_encode_normal(frac: float, fp8_exp: int, exp_bits: int, man_bits: int, sign: int) -> int:
    """Encode a normal FP8 value given the biased exponent and frexp fraction."""
    max_exp_raw = (1 << exp_bits) - 1
    max_man = (1 << man_bits) - 1
    man_int = round((2 * frac - 1) * (1 << man_bits))
    if man_int > max_man:
        # Rounding carried into exponent
        man_int = 0
        fp8_exp += 1
    if fp8_exp >= max_exp_raw:
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
    return (sign << 7) | (fp8_exp << man_bits) | man_int


def _float32_to_fp8_scalar(val: float, exp_bits: int) -> int:
    """Encode a float as one FP8 byte.

    Overflows clamp to the maximum finite value; NaN maps to a NaN encoding.
    """
    man_bits = 7 - exp_bits
    max_exp_raw = (1 << exp_bits) - 1  # all-ones exponent = NaN

    if math.isnan(val):
        return (max_exp_raw << man_bits) | 1  # canonical NaN

    sign = 0
    if val < 0 or (val == 0.0 and math.copysign(1.0, val) < 0):
        sign = 1
        val = abs(val)

    if val == 0.0:
        return sign << 7

    if math.isinf(val):
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)

    frac, exp = math.frexp(val)  # val = frac * 2^exp, 0.5 <= frac < 1
    bias = (1 << (exp_bits - 1)) - 1
    fp8_exp = (exp - 1) + bias  # biased exponent (exp-1 is the IEEE exponent)

    if fp8_exp >= max_exp_raw:
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
    if fp8_exp <= 0:
        return _fp8_encode_subnormal(val, exp_bits, man_bits, sign)
    return _fp8_encode_normal(frac, fp8_exp, exp_bits, man_bits, sign)


def fp32_to_fp8_bytes(fp32_values: np.ndarray, dtype: DType) -> bytes:
    """Convert an array of FP32 values to FP8 bytes.

    Uses the generic e(x)m(8-x) encoder; dtype must not be INT8.
    """
    if dtype == DType.INT8:
        raise ValueError(f"fp32_to_fp8_bytes only supports FP8 dtypes, got {dtype!r}")
    exp_bits = int(dtype)
    arr = fp32_values.astype(np.float32)
    return bytes(_float32_to_fp8_scalar(float(v), exp_bits) for v in arr)


def fp8_bytes_to_fp32(raw: bytes, dtype: DType) -> np.ndarray:
    """Convert raw FP8 bytes back to an FP32 numpy array."""
    if dtype == DType.INT8:
        raise ValueError(f"fp8_bytes_to_fp32 only supports FP8 dtypes, got {dtype!r}")
    exp_bits = int(dtype)
    return np.array(
        [_fp8_to_float32_scalar(b, exp_bits) for b in raw], dtype=np.float32
    )


def dtype_one_byte(dtype: DType) -> int:
    """Return the raw uint8 encoding of the value 1 for the given dtype.

    Used to pad out-of-bounds RC elements in mult.ve.cr / mult.ve.aaq.
    """
    if dtype == DType.INT8:
        return 0x01
    return _float32_to_fp8_scalar(1.0, int(dtype))


def ipu_mult(a_byte: int, b_byte: int, dtype: int) -> int | float:
    """Multiply two bytes according to *dtype*, returning int32 or float.

    Mirrors ``ipu_math__mult`` from C.
    """
    if dtype == DType.INT8:
        return _int8_to_signed(a_byte) * _int8_to_signed(b_byte)
    exp_bits = int(dtype)
    return _fp8_to_float32_scalar(a_byte, exp_bits) * _fp8_to_float32_scalar(b_byte, exp_bits)


def ipu_add(a_word: int | float, b_word: int | float, dtype: int) -> int | float:
    """Add two accumulator-width values according to *dtype*.

    Mirrors ``ipu_math__add`` from C.
    """
    if dtype == DType.INT8:
        # Both are int32 — wrap to 32 bits with two's complement
        result = (a_word + b_word) & 0xFFFFFFFF
        return result - 0x100000000 if result >= 0x80000000 else result
    else:
        # FP8 variants: accumulator is float
        return a_word + b_word
