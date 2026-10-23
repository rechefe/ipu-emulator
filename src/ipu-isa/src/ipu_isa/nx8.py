"""NX8: the hardware's 8-bit element family with a per-row scale (ADR 0007).

An NX8 row is 128 elements of at most 8 bits plus a metadata word holding the
row's ``scale`` (``e8m0``, bias 127) and, for a dynamic format, the
``dynamic_exponent`` split. Each element stores its exponent as a distance
*below* the scale, so one byte means different values in different rows.

:func:`encode_row` is the AaQ quantizer (``ARCH-stage-aaq-str.md`` §4.3.2,
including the dynamic rule of §4.3.2.3); :func:`decode_element` is the MULT
expansion (``ARCH-stage-mult.md`` §3.2.1), which inverts it. Both work on the
FP32 bit fields with integer arithmetic, so they are exact. The emulator and
the kernel harness share them.

Choices the specs leave open (ADR 0007, OQ-36 to OQ-43): ±inf saturates to the
largest magnitude and is left out of the scale; a NaN input raises; a negative
input to an unsigned format encodes as 0. Every saturation is reported, for the
``QUANT_SATURATION`` arithmetic event. FP32 inputs are "always treated as
normalized" (§4.3.2): a zero or subnormal gets an implicit leading 1.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from ipu_isa.errors import IsaValueError

#: The ``e8m0`` bias of the scale.
SCALE_BIAS: Final = 127
#: Elements per row.
ROW_ELEMENTS: Final = 128
_FP32_MANTISSA_BITS: Final = 23


@dataclass(frozen=True)
class Nx8Format:
    """One NX8 format: the fields of a QF half-CR (ADR 0007).

    ``exp_format`` is the exponent width ``fe`` of a static format; a dynamic
    format chooses ``fe`` = 1 or 2 per row and must be signed and 8 bits wide.
    """

    sign: int
    exp_format: int
    num_bits: int
    dynamic: bool = False

    def __post_init__(self) -> None:
        reason = self.invalid_reason()
        if reason is not None:
            raise IsaValueError(f"invalid NX8 format {self}: {reason}")

    def invalid_reason(self) -> str | None:
        """Why this format is invalid, or ``None``."""
        if self.sign not in (0, 1):
            return "sign must be 0 or 1"
        if not 1 <= self.num_bits <= 8:
            return "num_bits must be 1-8"
        if self.dynamic:
            if self.sign != 1 or self.num_bits != 8:
                return "a dynamic format is signed and 8 bits wide"
            return None
        if not 1 <= self.exp_format <= 7:
            return "exp_format (fe) must be 1-7"
        if self.exp_format + self.sign > self.num_bits:
            return "fe + sign exceeds num_bits"
        return None

    def exp_bits(self, dynamic_exponent: int = 0) -> int:
        """``fe`` for a row: the static width, or 1/2 from ``dynamic_exponent``."""
        if self.dynamic:
            return 2 if dynamic_exponent else 1
        return self.exp_format

    def man_bits(self, dynamic_exponent: int = 0) -> int:
        """``fm = num_bits - fe - sign``."""
        return self.num_bits - self.exp_bits(dynamic_exponent) - self.sign


@dataclass(frozen=True)
class EncodedRow:
    """One quantized row: 128 element bytes and its metadata fields."""

    elements: bytes
    scale: int
    dynamic_exponent: int
    #: True when any element saturated (±inf, edge 1 at E = 0, or a negative
    #: input to an unsigned format): the ``QUANT_SATURATION`` event.
    saturated: bool


@dataclass(frozen=True)
class _Fp32:
    sign: int
    exp: int  # raw biased exponent; 0 for zero and subnormals
    sig: int  # 24-bit significand with the implicit 1 ("treated as normalized")
    inf: bool
    zero: bool


def _fp32_fields(value: float) -> _Fp32:
    if math.isnan(value):
        raise IsaValueError("NX8 has no NaN encoding (OQ-37)")
    (bits,) = struct.unpack(">I", struct.pack(">f", value))
    sign = bits >> 31
    exp = (bits >> _FP32_MANTISSA_BITS) & 0xFF
    if exp == 0xFF or math.isinf(value):
        return _Fp32(sign, 0xFF, 0, True, False)
    frac = bits & 0x7FFFFF
    return _Fp32(sign, exp, frac | (1 << _FP32_MANTISSA_BITS), False, exp == 0 and frac == 0)


def _round_shift(value: int, shift: int) -> int:
    """``RTN(value / 2^shift)``, an exact half rounding up (§4.3.2.2)."""
    if shift <= 0:
        return value << -shift
    return (value + (1 << (shift - 1))) >> shift


def _encode_fields(p: _Fp32, scale: int, fe: int, fm: int) -> tuple[int, int, bool]:
    """``(Exp, M, saturated)`` of one finite element (§4.3.2.2)."""
    clamp_exp = (1 << fe) - 1
    distance = scale - p.exp
    if distance < clamp_exp:  # direct branch
        man = _round_shift(p.sig, _FP32_MANTISSA_BITS - fm) - (1 << fm)
        if man == 1 << fm:  # rounding carried out of fm bits
            if distance > 0:
                return distance - 1, 0, False
            return 0, (1 << fm) - 1, True  # no binade above the scale: saturate
        return distance, man, False
    shift = distance - clamp_exp + 1  # clamp branch: keep the magnitude
    man = _round_shift(p.sig, _FP32_MANTISSA_BITS - fm + shift)
    if man == 1 << fm:
        return clamp_exp - 1, 0, False
    return clamp_exp, man, False


def _pack(sign: int, exp: int, man: int, fmt: Nx8Format, fe: int, fm: int) -> int:
    code = (exp << fm) | man
    if fmt.sign:
        code |= sign << (fe + fm)
    return code << (8 - fmt.num_bits)


def _active_list(active: Sequence[bool] | None, n: int) -> list[bool]:
    if active is None:
        return [True] * n
    if len(active) != n:
        raise IsaValueError(f"active mask has {len(active)} entries for {n} values")
    return list(active)


def row_scale(values: Sequence[float], active: Sequence[bool] | None = None) -> int:
    """The row's scale: the largest raw exponent among active finite elements (§4.3.2.1)."""
    flags = _active_list(active, len(values))
    exps = [p.exp for p, on in zip(map(_fp32_fields, values), flags, strict=True) if on and not p.inf]
    return max(exps, default=0)


def _magnitude(p: _Fp32) -> Fraction:
    return Fraction(p.sig) * Fraction(2) ** (p.exp - SCALE_BIAS - _FP32_MANTISSA_BITS)


def _truncation_error(mag: Fraction, step: Fraction) -> Fraction:
    return mag - (mag // step) * step


def _rtn(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def dynamic_scores(
    values: Sequence[float], num_bits: int = 8, active: Sequence[bool] | None = None
) -> tuple[int, int]:
    """``(sum(A), sum(B))`` of the dynamic rule (§4.3.2.3), for a signed width.

    ``num_bits`` is a parameter only so the spec's worked example (``W = 6``)
    can be checked; a dynamic format itself is 8 bits wide (OQ-43). Errors are
    truncation errors with the steps of the §4.3.2.3 table, as the example uses.
    """
    fm1 = num_bits - 2
    flags = _active_list(active, len(values))
    fields = [_fp32_fields(v) for v in values]
    scale = max((p.exp for p, on in zip(fields, flags, strict=True) if on and not p.inf), default=0)
    unbiased = scale - SCALE_BIAS
    two = Fraction(2)
    sum_a = sum_b = 0
    for p, on in zip(fields, flags, strict=True):
        if not on or p.inf:
            continue
        distance = scale - p.exp
        if distance == 0:
            weight = two ** (2 * fm1)
        elif 1 < distance <= fm1 + 2:
            weight = two ** (2 * (fm1 + 1))
        else:
            continue
        mag = _magnitude(p)
        err1 = _truncation_error(mag, two ** (unbiased - min(distance, 1) - fm1))
        err2 = _truncation_error(mag, two ** (unbiased - min(distance, 3) - fm1 + 1))
        term = _rtn(abs(err2 * err2 - err1 * err1) * weight)
        if distance == 0:
            sum_a += term
        else:
            sum_b += term
    return sum_a, sum_b


def choose_dynamic_exponent(values: Sequence[float], active: Sequence[bool] | None = None) -> int:
    """``dynamic_exponent`` for a dynamic row: 0 (E1M6) or 1 (E2M5), §4.3.2.3."""
    sum_a, sum_b = dynamic_scores(values, 8, active)
    return 0 if sum_b - 4 * sum_a < 0 else 1


def encode_row(
    values: Sequence[float], fmt: Nx8Format, active: Sequence[bool] | None = None
) -> EncodedRow:
    """Quantize up to 128 FP32 values to one NX8 row (§4.3.2).

    Inactive elements (and the missing tail of a short row) are written as 0
    (OQ-36). Values are rounded to FP32 first, as they come from ``R_ACC``.
    """
    if len(values) > ROW_ELEMENTS:
        raise IsaValueError(f"an NX8 row holds {ROW_ELEMENTS} elements, got {len(values)}")
    flags = _active_list(active, len(values))
    fields = [_fp32_fields(v) for v in values]
    scale = max((p.exp for p, on in zip(fields, flags, strict=True) if on and not p.inf), default=0)
    dyn = choose_dynamic_exponent(values, active) if fmt.dynamic else 0
    fe, fm = fmt.exp_bits(dyn), fmt.man_bits(dyn)
    out = bytearray(ROW_ELEMENTS)
    saturated = False
    for i, (p, on) in enumerate(zip(fields, flags, strict=True)):
        if not on:
            continue
        if not fmt.sign and p.sign and not p.zero:
            exp, man, sat = (1 << fe) - 1, 0, True  # negative into unsigned: 0
        elif p.inf:
            exp, man, sat = 0, (1 << fm) - 1, True
        else:
            exp, man, sat = _encode_fields(p, scale, fe, fm)
        saturated |= sat
        out[i] = _pack(p.sign, exp, man, fmt, fe, fm)
    return EncodedRow(bytes(out), scale, dyn, saturated)


def decode_element(code: int, fmt: Nx8Format, scale: int, dynamic_exponent: int = 0) -> float:
    """The FP32 value of one NX8 element (``ARCH-stage-mult.md`` §3.2.1)."""
    if not 0 <= code <= 0xFF:
        raise IsaValueError(f"element code {code} is not a byte")
    fe, fm = fmt.exp_bits(dynamic_exponent), fmt.man_bits(dynamic_exponent)
    bits = code >> (8 - fmt.num_bits)
    man = bits & ((1 << fm) - 1)
    exp = (bits >> fm) & ((1 << fe) - 1)
    sign = (bits >> (fe + fm)) & 1 if fmt.sign else 0
    clamp_exp = (1 << fe) - 1
    unbiased = scale - SCALE_BIAS
    if exp < clamp_exp:
        mag = math.ldexp(1 + man / (1 << fm), unbiased - exp)
    else:
        mag = math.ldexp(man / (1 << fm), unbiased - (clamp_exp - 1))
    return -mag if sign else mag


def decode_row(
    elements: bytes, fmt: Nx8Format, scale: int, dynamic_exponent: int = 0
) -> list[float]:
    """:func:`decode_element` of every byte of *elements*."""
    return [decode_element(b, fmt, scale, dynamic_exponent) for b in elements]
