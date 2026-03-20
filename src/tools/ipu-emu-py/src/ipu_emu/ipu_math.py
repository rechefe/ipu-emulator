"""IPU math operations — Python port of ipu_math.c.

Uses ``ml_dtypes`` for FP8 E4M3 / E5M2 conversions, giving numpy-native
support that matches the hardware semantics exactly.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from ml_dtypes import float8_e4m3fn, float8_e5m2


class DType(IntEnum):
    """Data type enumeration — matches ``ipu_math__dtype_t`` in C."""
    INT8 = 0
    FP8_E4M3 = 1
    FP8_E5M2 = 2


# Lookup from DType to the ml_dtypes numpy dtype used for FP8 conversion.
_FP8_NUMPY_DTYPE = {
    DType.FP8_E4M3: float8_e4m3fn,
    DType.FP8_E5M2: float8_e5m2,
}


def _int8_to_signed(val: int) -> int:
    """Interpret an unsigned byte as a signed int8."""
    return val if val < 128 else val - 256


def _fp8_to_float(val: int, dtype: DType) -> float:
    """Convert a raw FP8 byte to a Python float via ml_dtypes."""
    np_dtype = _FP8_NUMPY_DTYPE[dtype]
    # Create a uint8 scalar, then view it as the FP8 dtype
    raw = np.array(val, dtype=np.uint8)
    return float(raw.view(np_dtype))


def fp32_to_fp8_bytes(fp32_values: np.ndarray, dtype: DType) -> bytes:
    """Convert an array of FP32 values to FP8 bytes.

    Uses ``ml_dtypes`` for exact hardware-matching conversion.
    """
    np_dtype = _FP8_NUMPY_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"fp32_to_fp8_bytes only supports FP8 dtypes, got {dtype!r}")
    fp8_arr = fp32_values.astype(np.float32).astype(np_dtype)
    return fp8_arr.view(np.uint8).tobytes()


def fp8_bytes_to_fp32(raw: bytes, dtype: DType) -> np.ndarray:
    """Convert raw FP8 bytes back to an FP32 numpy array."""
    np_dtype = _FP8_NUMPY_DTYPE.get(dtype)
    if np_dtype is None:
        raise ValueError(f"fp8_bytes_to_fp32 only supports FP8 dtypes, got {dtype!r}")
    arr = np.frombuffer(raw, dtype=np.uint8).view(np_dtype)
    return arr.astype(np.float32)


def ipu_mult(a_byte: int, b_byte: int, dtype: int) -> int | float:
    """Multiply two bytes according to *dtype*, returning int32 or float.

    Mirrors ``ipu_math__mult`` from C.
    """
    if dtype == DType.INT8:
        a = _int8_to_signed(a_byte)
        b = _int8_to_signed(b_byte)
        return a * b  # int32 result
    elif dtype in _FP8_NUMPY_DTYPE:
        return _fp8_to_float(a_byte, dtype) * _fp8_to_float(b_byte, dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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


# Raw byte value of the multiplicative identity "1" for each dtype.
# Used when a cyclic-register offset is out of bounds.
_DTYPE_ONE_BYTE: dict[int, int] = {
    DType.INT8: 0x01,    # signed int8: 1
    DType.FP8_E4M3: 0x38,  # float8_e4m3fn: 1.0  (0 0111 000)
    DType.FP8_E5M2: 0x3C,  # float8_e5m2:   1.0  (0 01111 00)
}


def dtype_one_byte(dtype: int) -> int:
    """Return the raw byte representing the multiplicative identity (1) for *dtype*.

    Used to pad out-of-bounds cyclic register accesses in mult.ve* instructions.

    Args:
        dtype: Data type (DType.INT8, DType.FP8_E4M3, or DType.FP8_E5M2).

    Returns:
        Raw uint8 byte value equal to 1 in the given dtype.

    Raises:
        ValueError: If *dtype* is not a supported data type.
    """
    try:
        return _DTYPE_ONE_BYTE[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}")
