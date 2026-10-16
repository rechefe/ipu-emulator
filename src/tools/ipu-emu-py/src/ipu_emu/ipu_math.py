"""IPU math operations — Python port of ipu_math.c.

The scalar adapters kept until E04-17 removes ``DType``: the element codec
lives in ``ipu_isa.formats`` and the lane codecs in ``ipu_emu.numerics``
(E04-13). ``ipu_add`` and ``ipu_sub`` stay the scalar accumulator arithmetic
the lane codecs are proven against.

Supports all e(x)m(8-x) 8-bit data types where x is the number of exponent
bits (0–7).  x=0 is the special integer mode (INT8); x=1..7 are FP8 formats
with IEEE-style encoding: 1 sign bit, x exponent bits, (7-x) mantissa bits,
and bias = 2^(x-1) - 1.
"""

from __future__ import annotations

import functools
from enum import IntEnum

import numpy as np

from ipu_emu.numerics import codec_for
from ipu_isa.formats import ElementFormat, decode_element, encode_element


class DType(IntEnum):
    """8-bit data type.

    The integer value equals the number of exponent bits (x).
    INT8 (x=0) is the integer mode; E1..E7 are FP8 formats e(x)m(8-x)
    where "m" counts sign + mantissa bits, giving 1 sign + x exponent
    + (7-x) stored mantissa bits = 8 bits total.
    """

    INT8 = 0  # integer mode: 8-bit signed (no exponent)
    E1 = 1  # fp8_e1: 1 exp bit,  6 mantissa bits
    E2 = 2  # fp8_e2: 2 exp bits, 5 mantissa bits
    E3 = 3  # fp8_e3: 3 exp bits, 4 mantissa bits
    E4 = 4  # fp8_e4: 4 exp bits, 3 mantissa bits  (formerly FP8_E4M3)
    E5 = 5  # fp8_e5: 5 exp bits, 2 mantissa bits  (formerly FP8_E5M2)
    E6 = 6  # fp8_e6: 6 exp bits, 1 mantissa bit
    E7 = 7  # fp8_e7: 7 exp bits, 0 mantissa bits


# DType value -> element format: DType(x) is the 8-bit format with x exponent bits.
_FORMAT_OF: dict[int, ElementFormat] = {int(d): ElementFormat(d.name.lower()) for d in DType}


def _format_of(dtype: int) -> ElementFormat:
    try:
        return _FORMAT_OF[int(dtype)]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype!r}") from None


def _fp8_to_float32_scalar(byte_val: int, exp_bits: int) -> float:
    """Decode one FP8 byte to a Python float (``ipu_isa.formats.decode_element``)."""
    return decode_element(byte_val, _format_of(exp_bits))


def _float32_to_fp8_scalar(val: float, exp_bits: int) -> int:
    """Encode a float as one FP8 byte (``ipu_isa.formats.encode_element``).

    Overflows clamp to the maximum finite value; NaN maps to a NaN encoding.
    """
    return encode_element(val, _format_of(exp_bits))


@functools.cache
def _element_values(dtype: int) -> tuple[int | float, ...]:
    """Every code's value as a Python int (INT8) or float, from the format's lane codec."""
    return tuple(codec_for(_format_of(dtype)).decode(bytes(range(256)), 256).tolist())


def fp32_to_fp8_bytes(fp32_values: np.ndarray, dtype: DType) -> bytes:
    """Convert an array of FP32 values to FP8 bytes.

    Uses the generic e(x)m(8-x) encoder; dtype must not be INT8.
    """
    if dtype == DType.INT8:
        raise ValueError(f"fp32_to_fp8_bytes only supports FP8 dtypes, got {dtype!r}")
    fmt = _format_of(dtype)
    arr = fp32_values.astype(np.float32)
    # NOTE: scalar Python loop — adequate for emulator use but not vectorized.
    return bytes(encode_element(float(v), fmt) for v in arr)


def fp8_bytes_to_fp32(raw: bytes, dtype: DType) -> np.ndarray:
    """Convert raw FP8 bytes back to an FP32 numpy array."""
    if dtype == DType.INT8:
        raise ValueError(f"fp8_bytes_to_fp32 only supports FP8 dtypes, got {dtype!r}")
    codec = codec_for(_format_of(dtype))
    return codec.decode(raw, len(raw)).astype(np.float32)


def dtype_one_byte(dtype: DType) -> int:
    """Return the raw uint8 encoding of the value 1 for the given dtype.

    Used to pad out-of-bounds RC elements in mult.ve.padded and in mult.ve.cr / mult.ve.aaq.
    """
    return codec_for(_format_of(dtype)).one()[0]


def ipu_mult(a_byte: int, b_byte: int, dtype: int) -> int | float:
    """Multiply two bytes according to *dtype*, returning int32 or float.

    Mirrors ``ipu_math__mult`` from C: the two elements decoded (the lane
    codec's table, as Python ints or floats) and multiplied as Python numbers.
    ``ipu_emu.numerics`` is the vectorised form; ``test_numerics`` holds the
    two bit-identical.
    """
    values = _element_values(dtype)
    return values[a_byte] * values[b_byte]


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


def ipu_sub(a_word: int | float, b_word: int | float, dtype: int) -> int | float:
    """Subtract b from a according to *dtype* (accumulator-width values)."""
    if dtype == DType.INT8:
        result = (a_word - b_word) & 0xFFFFFFFF
        return result - 0x100000000 if result >= 0x80000000 else result
    else:
        return a_word - b_word
