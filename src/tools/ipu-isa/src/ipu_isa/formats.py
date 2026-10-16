"""Element formats and every size derived from them (E03-13, decision Q47).

One enum names every element format: ``INT8`` and ``E1`` to ``E7`` are
HARDWARE (selected at run time); ``FP32`` and ``INT32`` exist only in the
emulator. Each format is authored once as data in ``_FORMAT_SPEC`` and loaded
into a frozen ``FormatInfo``. Element bytes, row bytes, the accumulator type,
the per-register element layout and the addressable XMEM rows are derived from
it, never restated.

The element codec of the 8-bit formats is here too (E04-13), so the harness
and the conformance tooling round and pack values without the emulator:
:func:`decode_element`, :func:`encode_element` and :func:`element_values`.
It is pure Python, one element at a time; the emulator's lane codecs
(``ipu_emu.numerics``) build their vectorised tables from it. The arithmetic
on accumulators belongs to those codecs, and the numeric questions still open
to OQ-19.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, TypedDict

from ipu_isa.geometry import LANES, XMEM_BYTES
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.schema import SpecError
from ipu_isa.tags import Tag


class ElementFormat(str, Enum):
    """An element format. ``ElementFormat.E4 == "e4"``.

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Tag``.
    """

    INT8 = "int8"
    E1 = "e1"
    E2 = "e2"
    E3 = "e3"
    E4 = "e4"
    E5 = "e5"
    E6 = "e6"
    E7 = "e7"
    FP32 = "fp32"
    INT32 = "int32"

    def __str__(self) -> str:
        return self.value


class AccumulatorType(str, Enum):
    """The type MULT_RES and R_ACC hold for a format (see GLOSSARY.md)."""

    INT32 = "int32"
    FP32 = "fp32"

    def __str__(self) -> str:
        return self.value


#: Bits of one accumulator element, per accumulator type.
ACCUMULATOR_BITS: Mapping[AccumulatorType, int] = MappingProxyType(
    {AccumulatorType.INT32: 32, AccumulatorType.FP32: 32}
)


@dataclass(frozen=True, slots=True)
class FormatInfo:
    """Everything the ISA states about one element format."""

    format: ElementFormat
    tag: Tag  # HARDWARE: INT8, E1..E7; EMULATOR: FP32, INT32
    is_float: bool
    element_bits: int  # 8 or 32
    exp_bits: int  # 0 for integers, x for Ex, 8 for FP32
    man_bits: int  # stored mantissa bits: 7 - x for Ex, 23 for FP32, 0 for integers
    accumulator: AccumulatorType  # integers -> INT32, every float -> FP32

    @property
    def element_bytes(self) -> int:
        """Bytes of one element."""
        return self.element_bits // 8

    @property
    def row_bytes(self) -> int:
        """Bytes of one XMEM row: one element per lane."""
        return LANES * self.element_bytes

    @property
    def accumulator_bytes(self) -> int:
        """Bytes of one MULT_RES / R_ACC element."""
        return ACCUMULATOR_BITS[self.accumulator] // 8


class _FormatEntry(TypedDict):
    tag: Tag
    is_float: bool
    element_bits: int
    exp_bits: int
    accumulator: AccumulatorType


# The authored data. Stored mantissa bits are derived: an element is one sign
# bit, ``exp_bits`` exponent bits and the rest mantissa (0 for integers).
_FORMAT_SPEC: Final[Mapping[ElementFormat, _FormatEntry]] = {
    ElementFormat.INT8: {
        "tag": Tag.HARDWARE,
        "is_float": False,
        "element_bits": 8,
        "exp_bits": 0,
        "accumulator": AccumulatorType.INT32,
    },
    **{
        ElementFormat(f"e{x}"): {
            "tag": Tag.HARDWARE,
            "is_float": True,
            "element_bits": 8,
            "exp_bits": x,
            "accumulator": AccumulatorType.FP32,
        }
        for x in range(1, 8)
    },
    ElementFormat.FP32: {
        "tag": Tag.EMULATOR,
        "is_float": True,
        "element_bits": 32,
        "exp_bits": 8,
        "accumulator": AccumulatorType.FP32,
    },
    ElementFormat.INT32: {
        "tag": Tag.EMULATOR,
        "is_float": False,
        "element_bits": 32,
        "exp_bits": 0,
        "accumulator": AccumulatorType.INT32,
    },
}


def _load(fmt: ElementFormat, entry: _FormatEntry) -> FormatInfo:
    man_bits = entry["element_bits"] - 1 - entry["exp_bits"] if entry["is_float"] else 0
    return FormatInfo(format=fmt, man_bits=man_bits, **entry)


def _bit_layout_error(info: FormatInfo) -> str | None:
    """Why ``info``'s bit layout or accumulator breaks a rule, or ``None``."""
    if info.element_bits <= 0 or info.element_bits % 8:
        return f".element_bits: {info.element_bits} is not a whole number of bytes"
    if not info.is_float:
        if info.exp_bits or info.man_bits:
            return ": an integer format has no exponent or mantissa bits"
        expected_acc = AccumulatorType.INT32
    else:
        if info.exp_bits < 1 or info.man_bits < 0:
            return ": a float format needs exponent bits"
        if 1 + info.exp_bits + info.man_bits != info.element_bits:
            return (
                f": 1 sign + {info.exp_bits} exponent + {info.man_bits} mantissa "
                f"bits != {info.element_bits}"
            )
        expected_acc = AccumulatorType.FP32
    if info.accumulator is not expected_acc:
        return f".accumulator: {info.accumulator}, expected {expected_acc}"
    return None


def validate_formats(formats: Mapping[ElementFormat, FormatInfo]) -> None:
    """Raise ``SpecError`` if a format breaks a schema rule.

    Every format is listed once with exactly one tag (HARDWARE or EMULATOR);
    a float format is ``1 + exp_bits + man_bits`` bits wide; FP32 has the IEEE
    binary32 layout; an integer format has no exponent or mantissa bits; the
    accumulator is INT32 for integer formats and FP32 for float formats.
    """
    missing = set(ElementFormat) - set(formats)
    if missing:
        raise SpecError(f"formats: missing {sorted(str(f) for f in missing)}")
    for fmt, info in formats.items():
        path = f"formats.{fmt}"
        if info.format is not fmt:
            raise SpecError(f"{path}: keyed under {fmt} but describes {info.format}")
        if info.tag not in (Tag.HARDWARE, Tag.EMULATOR):
            raise SpecError(f"{path}.tag: {info.tag} is not HARDWARE or EMULATOR")
        error = _bit_layout_error(info)
        if error is not None:
            raise SpecError(path + error)
    fp32 = formats[ElementFormat.FP32]
    if (fp32.element_bits, fp32.exp_bits, fp32.man_bits) != (32, 8, 23):
        raise SpecError("formats.fp32: not the IEEE binary32 layout (8 exponent, 23 mantissa bits)")


#: Every element format, in enum order.
FORMATS: Mapping[ElementFormat, FormatInfo] = MappingProxyType(
    {fmt: _load(fmt, _FORMAT_SPEC[fmt]) for fmt in ElementFormat}
)
validate_formats(FORMATS)


# How each register's elements depend on the format. Element registers hold
# one element per byte of their 8-bit-format allocation, at the format's
# width; accumulator registers hold one accumulator element per lane; R_MASK
# holds lane-mask bits and is the same in every format.
_ELEMENT_REGISTERS: Final = frozenset({"r", "r_cyclic", "mem_bypass"})
_ACCUMULATOR_REGISTERS: Final = frozenset({"mult_res", "r_acc"})
_MASK_REGISTERS: Final = frozenset({"r_mask"})
_NARROW_ELEMENT_BYTES: Final = 1


def register_layout(register: str, fmt: ElementFormat) -> tuple[int, int]:
    """Return ``(elements, element_bytes)`` of one ``register`` in format ``fmt``.

    ``register`` is a ``REGISTER_DEFINITIONS`` key. This is what the emulator
    must allocate per format (E04-17); ``REGISTER_DEFINITIONS`` itself still
    holds the 8-bit-format sizes.
    """
    info = FORMATS[ElementFormat(fmt)]
    if register not in REGISTER_DEFINITIONS:
        raise KeyError(f"unknown register {register!r}")
    size_bytes = REGISTER_DEFINITIONS[register]["size_bytes"]
    if register in _ELEMENT_REGISTERS:
        return size_bytes // _NARROW_ELEMENT_BYTES, info.element_bytes
    if register in _ACCUMULATOR_REGISTERS:
        return LANES, info.accumulator_bytes
    if register in _MASK_REGISTERS:
        return size_bytes, 1
    raise KeyError(f"register {register!r} has no format-dependent element layout")


def addressable_rows(fmt: ElementFormat) -> int:
    """XMEM rows a program may address in ``fmt``.

    The byte address space (``XMEM_BYTES``) is the same in every format; the
    row count is that space divided by the row width. The emulator still
    limits 8-bit formats to 16384 rows until E04 adopts this; the hardware row
    limit is an open question.
    """
    return XMEM_BYTES // FORMATS[ElementFormat(fmt)].row_bytes


# --- the 8-bit element codec (E04-13) ----------------------------------------
#
# An Ex format (x = 1..7) is 1 sign bit, x exponent bits and 7 - x mantissa
# bits, with bias 2^(x-1) - 1. The all-ones exponent is NaN (no infinities);
# the zero exponent is subnormal. E1 is degenerate: bias 0, and its only
# non-zero exponent is the NaN exponent, so every finite value is subnormal.
# E7 has no mantissa bits: its magnitudes are powers of two. INT8 is a signed
# byte. Encoding rounds to nearest, ties to even, and saturates; NaN encodes
# as the canonical NaN.


def _eight_bit_info(fmt: ElementFormat) -> FormatInfo:
    info = FORMATS[ElementFormat(fmt)]
    if info.element_bits != 8:
        raise ValueError(f"{info.format} is not an 8-bit format")
    return info


def _fp8_max_finite(exp_bits: int, man_bits: int) -> int:
    """The code of the largest positive finite value."""
    max_exp_raw = (1 << exp_bits) - 1  # all-ones = NaN, so the largest normal is one below
    return ((max_exp_raw - 1) << man_bits) | ((1 << man_bits) - 1)


def _fp8_decode(code: int, exp_bits: int) -> float:
    man_bits = 7 - exp_bits
    sign = (code >> 7) & 1
    exp_raw = (code >> man_bits) & ((1 << exp_bits) - 1)
    man_raw = code & ((1 << man_bits) - 1)
    if exp_raw == (1 << exp_bits) - 1:
        return float("nan")
    bias = (1 << (exp_bits - 1)) - 1
    if exp_raw == 0:
        # Subnormal: 0.man * 2^(1-bias)
        value = (man_raw / (1 << man_bits)) * (2.0 ** (1 - bias))
    else:
        # Normal: 1.man * 2^(exp-bias)
        value = (1 + man_raw / (1 << man_bits)) * (2.0 ** (exp_raw - bias))
    return -value if sign else value


def _fp8_encode_subnormal(val: float, exp_bits: int, man_bits: int, sign: int) -> int:
    """Encode a positive magnitude below the smallest normal."""
    bias = (1 << (exp_bits - 1)) - 1
    max_man = (1 << man_bits) - 1
    # val = (man_int / 2^man_bits) * 2^(1-bias)  ->  man_int = val * 2^(man_bits+bias-1)
    man_int = round(val * (2.0 ** (man_bits + bias - 1)))
    if man_int > max_man:
        # Rounded up out of the subnormal range. (1 << man_bits) is exponent
        # field 1, mantissa 0: the smallest normal. E1 has no normal range
        # (exponent field 1 is NaN), so it saturates instead.
        if exp_bits == 1:
            return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
        man_int = 1 << man_bits
    return (sign << 7) | man_int


def _fp8_encode_normal(frac: float, fp8_exp: int, exp_bits: int, sign: int) -> int:
    """Encode a normal value given its biased exponent and frexp fraction."""
    man_bits = 7 - exp_bits
    max_exp_raw = (1 << exp_bits) - 1
    max_man = (1 << man_bits) - 1
    # man_int is in [0, 2^man_bits]; round() is round half to even, and since
    # the mantissa holds the low bits of the code, an even mantissa is an even
    # code. E7 (man_bits=0) has no mantissa bit to make even: a tie lies
    # halfway between two powers of two, and it goes to the even code, i.e.
    # it carries into the next exponent when fp8_exp is odd.
    scaled = (2 * frac - 1) * (1 << man_bits)
    man_int = round(scaled)
    if man_bits == 0 and scaled == 0.5 and (fp8_exp & 1):
        man_int = 1
    if man_int > max_man:
        # Rounding carried into the exponent.
        man_int = 0
        fp8_exp += 1
    if fp8_exp >= max_exp_raw:
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
    return (sign << 7) | (fp8_exp << man_bits) | man_int


def _fp8_encode(val: float, exp_bits: int) -> int:
    man_bits = 7 - exp_bits
    max_exp_raw = (1 << exp_bits) - 1
    if math.isnan(val):
        return (max_exp_raw << man_bits) | 1  # canonical NaN
    sign = 1 if val < 0 or (val == 0.0 and math.copysign(1.0, val) < 0) else 0
    val = abs(val)
    if val == 0.0:
        return sign << 7  # -0.0 encodes as 0x80
    if math.isinf(val):
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
    frac, exp = math.frexp(val)  # val = frac * 2^exp, 0.5 <= frac < 1
    bias = (1 << (exp_bits - 1)) - 1
    fp8_exp = (exp - 1) + bias  # biased exponent (exp-1 is the IEEE exponent)
    if fp8_exp >= max_exp_raw:
        return (sign << 7) | _fp8_max_finite(exp_bits, man_bits)
    if fp8_exp <= 0:
        return _fp8_encode_subnormal(val, exp_bits, man_bits, sign)
    return _fp8_encode_normal(frac, fp8_exp, exp_bits, sign)


def decode_element(code: int, fmt: ElementFormat) -> float:
    """The value of one element of an 8-bit format.

    Args:
        code: The element's byte, 0..255.
        fmt: ``INT8`` or one of ``E1``..``E7``.

    Returns:
        The value as a float: a signed integer for ``INT8``; NaN (positive,
        whatever the sign bit) for an all-ones exponent.

    Raises:
        ValueError: ``fmt`` is not an 8-bit format, or ``code`` is not a byte.
    """
    info = _eight_bit_info(fmt)
    if not 0 <= code <= 0xFF:
        raise ValueError(f"element code {code} is not a byte")
    if not info.is_float:
        return float(code - 0x100 if code & 0x80 else code)
    return _fp8_decode(code, info.exp_bits)


def encode_element(value: float, fmt: ElementFormat) -> int:
    """The byte of the 8-bit-format element nearest to ``value``.

    Rounds to nearest, ties to even, and saturates to the largest finite
    value (infinities too). An Ex NaN encodes as the canonical NaN (exponent
    all ones, mantissa 1); ``-0.0`` keeps its sign. ``INT8`` rounds to the
    nearest integer and saturates to -128..127; it has no NaN.

    Args:
        value: The value to encode.
        fmt: ``INT8`` or one of ``E1``..``E7``.

    Returns:
        The element's byte, 0..255.

    Raises:
        ValueError: ``fmt`` is not an 8-bit format, or ``value`` is NaN and
            ``fmt`` is ``INT8``.
    """
    info = _eight_bit_info(fmt)
    if info.is_float:
        return _fp8_encode(float(value), info.exp_bits)
    if math.isnan(value):
        raise ValueError("NaN has no INT8 encoding")
    clipped = min(127.0, max(-128.0, float(value)))
    return round(clipped) & 0xFF


@functools.cache
def element_values(fmt: ElementFormat) -> tuple[float, ...]:
    """Every code's value of an 8-bit format: entry ``c`` is ``decode_element(c, fmt)``.

    Args:
        fmt: ``INT8`` or one of ``E1``..``E7``.

    Returns:
        256 floats, indexed by code (the decode table).

    Raises:
        ValueError: ``fmt`` is not an 8-bit format.
    """
    fmt = _eight_bit_info(fmt).format
    return tuple(decode_element(code, fmt) for code in range(256))
