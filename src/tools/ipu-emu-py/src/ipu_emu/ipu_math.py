"""IPU math operations — Python port of ipu_math.c.

Scalar adapters over the element formats of ``ipu_isa.formats``: the element
codec lives there and the lane codecs in ``ipu_emu.numerics`` (E04-13).
``ipu_add`` and ``ipu_sub`` stay the scalar accumulator arithmetic the lane
codecs are proven against.

The 8-bit formats are INT8 (signed integer) and E1..E7, the e(x)m(8-x)
floating-point formats with IEEE-style encoding: 1 sign bit, x exponent bits,
(7-x) mantissa bits, and bias = 2^(x-1) - 1.
"""

from __future__ import annotations

import functools

import numpy as np

from ipu_emu.numerics import codec_for
from ipu_isa.formats import FORMATS, ElementFormat, decode_element, encode_element


def _check_fp8(fmt: ElementFormat, what: str) -> ElementFormat:
    info = FORMATS.get(ElementFormat(fmt))
    if info is None or not info.is_float or info.element_bits != 8:
        raise ValueError(f"{what} only supports the FP8 formats E1..E7, got {fmt!r}")
    return info.format


# Exponent bits -> 8-bit format: 0 is INT8, x is Ex.
_FORMAT_OF_EXP_BITS: dict[int, ElementFormat] = {
    info.exp_bits: fmt for fmt, info in FORMATS.items() if info.element_bits == 8
}


def _format_of(exp_bits: int) -> ElementFormat:
    try:
        return _FORMAT_OF_EXP_BITS[int(exp_bits)]
    except KeyError:
        raise ValueError(f"no 8-bit format has {exp_bits!r} exponent bits") from None


def _fp8_to_float32_scalar(byte_val: int, exp_bits: int) -> float:
    """Decode one FP8 byte to a Python float (``ipu_isa.formats.decode_element``)."""
    return decode_element(byte_val, _format_of(exp_bits))


def _float32_to_fp8_scalar(val: float, exp_bits: int) -> int:
    """Encode a float as one FP8 byte (``ipu_isa.formats.encode_element``).

    Overflows clamp to the maximum finite value; NaN maps to a NaN encoding.
    """
    return encode_element(val, _format_of(exp_bits))


@functools.cache
def _element_values(fmt: ElementFormat) -> tuple[int | float, ...]:
    """Every code's value as a Python int (INT8) or float, from the format's lane codec."""
    return tuple(codec_for(fmt).decode(bytes(range(256)), 256).tolist())


def fp32_to_fp8_bytes(fp32_values: np.ndarray, fmt: ElementFormat) -> bytes:
    """Convert an array of FP32 values to FP8 bytes in *fmt* (one of E1..E7).

    Raises:
        ValueError: *fmt* is not an FP8 format.
    """
    fmt = _check_fp8(fmt, "fp32_to_fp8_bytes")
    arr = fp32_values.astype(np.float32)
    # NOTE: scalar Python loop — adequate for emulator use but not vectorized.
    return bytes(encode_element(float(v), fmt) for v in arr)


def fp8_bytes_to_fp32(raw: bytes, fmt: ElementFormat) -> np.ndarray:
    """Convert raw FP8 bytes in *fmt* (one of E1..E7) back to an FP32 numpy array.

    Raises:
        ValueError: *fmt* is not an FP8 format.
    """
    fmt = _check_fp8(fmt, "fp8_bytes_to_fp32")
    codec = codec_for(fmt)
    return codec.decode(raw, len(raw)).astype(np.float32)


def _is_integer(fmt: ElementFormat) -> bool:
    return not FORMATS[ElementFormat(fmt)].is_float


def ipu_mult(a_byte: int, b_byte: int, fmt: ElementFormat) -> int | float:
    """Multiply two bytes in the 8-bit format *fmt*, returning int32 or float.

    Mirrors ``ipu_math__mult`` from C: the two elements decoded (the lane
    codec's table, as Python ints or floats) and multiplied as Python numbers.
    ``ipu_emu.numerics`` is the vectorised form; ``test_numerics`` holds the
    two bit-identical.
    """
    values = _element_values(ElementFormat(fmt))
    return values[a_byte] * values[b_byte]


def ipu_add(a_word: int | float, b_word: int | float, fmt: ElementFormat) -> int | float:
    """Add two accumulator-width values in the accumulator of *fmt*.

    Mirrors ``ipu_math__add`` from C.
    """
    if _is_integer(fmt):
        # Both are int32 — wrap to 32 bits with two's complement
        result = (a_word + b_word) & 0xFFFFFFFF
        return result - 0x100000000 if result >= 0x80000000 else result
    else:
        # Float formats: the accumulator is float
        return a_word + b_word


def ipu_sub(a_word: int | float, b_word: int | float, fmt: ElementFormat) -> int | float:
    """Subtract b from a in the accumulator of *fmt* (accumulator-width values)."""
    if _is_integer(fmt):
        result = (a_word - b_word) & 0xFFFFFFFF
        return result - 0x100000000 if result >= 0x80000000 else result
    else:
        return a_word - b_word
