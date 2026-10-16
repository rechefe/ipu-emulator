"""Exhaustive tests for the host-side FP8 codec in ``ipu_math``.

Every expected value comes from an independent reference built from the
format definition with ``fractions.Fraction`` (or from ``ml_dtypes``), never
from the encoder under test and never from a private ``ipu_math`` helper.

Format e(x): 1 sign bit, x exponent bits, (7 - x) mantissa bits,
bias 2**(x-1) - 1.  The all-ones exponent is NaN (no infinities); the zero
exponent is subnormal.  Overflow saturates to the largest finite value.
"""

from __future__ import annotations

import bisect
import math
from fractions import Fraction

import ml_dtypes
import numpy as np
import pytest

from ipu_emu.ipu_math import (
    DType,
    _float32_to_fp8_scalar,
    fp8_bytes_to_fp32,
    fp32_to_fp8_bytes,
)

FORMATS = [DType.E1, DType.E2, DType.E3, DType.E4, DType.E5, DType.E6, DType.E7]


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------


def _fields(e: int) -> tuple[int, int]:
    """Return (mantissa bits, bias) of format e(e)."""
    return 7 - e, 2 ** (e - 1) - 1


def ref_is_nan(code: int, e: int) -> bool:
    m, _ = _fields(e)
    return ((code >> m) & (2**e - 1)) == 2**e - 1


def ref_decode(code: int, e: int) -> Fraction | None:
    """Exact value of *code* in format e(e); None for NaN.

    The sign of zero is not represented by a Fraction; callers that care
    about it check the sign bit directly.
    """
    m, bias = _fields(e)
    sign = -1 if code & 0x80 else 1
    exp_field = (code >> m) & (2**e - 1)
    man_field = code & (2**m - 1)
    if exp_field == 2**e - 1:
        return None
    if exp_field == 0:
        mag = Fraction(man_field, 2**m) * Fraction(2) ** (1 - bias)
    else:
        mag = (1 + Fraction(man_field, 2**m)) * Fraction(2) ** (exp_field - bias)
    return sign * mag


def ref_positive_table(e: int) -> list[tuple[Fraction, int]]:
    """All non-negative finite (magnitude, code) pairs, ascending."""
    table = [(ref_decode(c, e), c) for c in range(0x80) if not ref_is_nan(c, e)]
    table.sort()
    # Magnitudes are strictly increasing with the code.
    assert [c for _, c in table] == list(range(len(table)))
    return table


_TABLES = {e: ref_positive_table(e) for e in range(1, 8)}


def ref_encode(x: float, e: int) -> int | None:
    """Nearest representable code to *x*, ties to the even code.

    Magnitudes above the largest finite value saturate to it (infinities
    included).  Returns None for NaN input (any NaN code is acceptable).
    """
    if math.isnan(x):
        return None
    sign = 0x80 if math.copysign(1.0, x) < 0 else 0
    table = _TABLES[e]
    mags = [v for v, _ in table]
    max_mag, max_code = table[-1]
    if math.isinf(x):
        return sign | max_code
    a = abs(Fraction(x))
    if a >= max_mag:
        return sign | max_code
    i = bisect.bisect_left(mags, a)
    if mags[i] == a:
        return sign | table[i][1]
    lo_v, lo_c = table[i - 1]
    hi_v, hi_c = table[i]
    d_lo, d_hi = a - lo_v, hi_v - a
    if d_lo < d_hi:
        code = lo_c
    elif d_hi < d_lo:
        code = hi_c
    else:
        code = lo_c if lo_c % 2 == 0 else hi_c
    return sign | code


def _f32(x) -> float:
    return float(np.float32(x))


def _next_up(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(np.inf)))


def _next_down(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(-np.inf)))


def _midpoint_inputs(e: int) -> list[float]:
    """Every midpoint of adjacent magnitudes, +/- one float32 ULP, both signs."""
    table = _TABLES[e]
    out = []
    for (lo, _), (hi, _) in zip(table, table[1:]):
        mid = (lo + hi) / 2
        f = _f32(float(mid))
        assert Fraction(f) == mid, "midpoint must be exact in float32"
        for v in (f, _next_up(f), _next_down(f)):
            out.extend((v, -v))
    return out


def _check_encode(x: float, e: int) -> None:
    got = _float32_to_fp8_scalar(x, e)
    want = ref_encode(x, e)
    if want is None:
        assert ref_is_nan(got, e), f"e{e}: {x!r} -> {got:#04x}, want NaN"
    else:
        assert got == want, f"e{e}: {x!r} -> {got:#04x}, want {want:#04x}"


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FORMATS, ids=lambda d: d.name)
def test_decode_exhaustive(dtype):
    e = int(dtype)
    got = fp8_bytes_to_fp32(bytes(range(256)), dtype)
    for code in range(256):
        want = ref_decode(code, e)
        if want is None:
            assert math.isnan(got[code]), f"{code:#04x}"
        else:
            assert Fraction(float(got[code])) == want, f"{code:#04x}"
            if want == 0:
                assert math.copysign(1.0, got[code]) == (-1.0 if code & 0x80 else 1.0)


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", FORMATS, ids=lambda d: d.name)
def test_encode_round_trip(dtype):
    e = int(dtype)
    for code in range(256):
        if ref_is_nan(code, e):
            continue
        value = ref_decode(code, e)
        x = float(value)
        if value == 0 and code & 0x80:
            x = -0.0
        assert Fraction(_f32(x)) == value
        assert _float32_to_fp8_scalar(x, e) == code, f"{code:#04x}"


@pytest.mark.parametrize("dtype", FORMATS, ids=lambda d: d.name)
def test_encode_midpoints(dtype):
    e = int(dtype)
    table = _TABLES[e]
    for (lo, lo_c), (hi, hi_c) in zip(table, table[1:]):
        mid = _f32(float((lo + hi) / 2))
        even = lo_c if lo_c % 2 == 0 else hi_c
        assert _float32_to_fp8_scalar(mid, e) == even
        assert _float32_to_fp8_scalar(-mid, e) == 0x80 | even
        assert _float32_to_fp8_scalar(_next_down(mid), e) == lo_c
        assert _float32_to_fp8_scalar(_next_up(mid), e) == hi_c
    for x in _midpoint_inputs(e):
        _check_encode(x, e)


@pytest.mark.parametrize("dtype", FORMATS[1:], ids=lambda d: d.name)  # E1 has no normal range
def test_encode_subnormal_to_normal_gap_dense(dtype):
    """Every float32 sample in [max_subnormal, min_normal) rounds to nearest."""
    e = int(dtype)
    m, _ = _fields(e)
    max_sub = float(ref_decode(2**m - 1, e))
    min_norm = float(ref_decode(2**m, e))
    for x in np.linspace(max_sub, min_norm, 4097, dtype=np.float32)[:-1]:
        _check_encode(float(x), e)
        _check_encode(-float(x), e)


@pytest.mark.parametrize("dtype", FORMATS, ids=lambda d: d.name)
def test_encode_saturation_and_nan(dtype):
    e = int(dtype)
    max_mag, max_code = _TABLES[e][-1]
    big = _f32(float(max_mag) * 1.01)
    assert _float32_to_fp8_scalar(big, e) == max_code
    assert _float32_to_fp8_scalar(-big, e) == 0x80 | max_code
    assert _float32_to_fp8_scalar(math.inf, e) == max_code
    assert _float32_to_fp8_scalar(-math.inf, e) == 0x80 | max_code
    nan_code = _float32_to_fp8_scalar(math.nan, e)
    assert ref_is_nan(nan_code, e)
    assert math.isnan(fp8_bytes_to_fp32(bytes([nan_code]), dtype)[0])


# Values just above the midpoint of the gap between the largest subnormal and
# the smallest normal.  Before E00-01 the encoder clamped these to the largest
# subnormal (or to zero for E7).
_GAP_REGRESSION = [
    (DType.E2, 0x20),
    (DType.E3, 0x10),
    (DType.E4, 0x08),
    (DType.E5, 0x04),
    (DType.E6, 0x02),
    (DType.E7, 0x01),
]


@pytest.mark.parametrize(
    "dtype,expected", _GAP_REGRESSION, ids=lambda p: getattr(p, "name", hex(p))
)
def test_subnormal_carry_regression(dtype, expected):
    e = int(dtype)
    m, _ = _fields(e)
    max_sub = float(ref_decode(2**m - 1, e))
    min_norm = float(ref_decode(2**m, e))
    v = _f32((max_sub + min_norm) / 2 * 1.0001)
    assert _float32_to_fp8_scalar(v, e) == expected
    assert _float32_to_fp8_scalar(-v, e) == 0x80 | expected


def test_e1_carry_saturates():
    assert _float32_to_fp8_scalar(1.99, 1) == 0x3F
    assert _float32_to_fp8_scalar(-1.99, 1) == 0xBF


def _all_scalar_inputs(e: int) -> list[float]:
    max_mag = float(_TABLES[e][-1][0])
    vals = [float(ref_decode(c, e)) for c in range(256) if not ref_is_nan(c, e)]
    vals += [0.0, -0.0, math.inf, -math.inf, math.nan]
    vals += [_f32(max_mag * 1.01), -_f32(max_mag * 1.01)]
    vals += _midpoint_inputs(e)
    return vals


@pytest.mark.parametrize("dtype", FORMATS, ids=lambda d: d.name)
def test_array_path_matches_scalar(dtype):
    e = int(dtype)
    vals = _all_scalar_inputs(e)
    got = fp32_to_fp8_bytes(np.array(vals, dtype=np.float32), dtype)
    want = bytes(_float32_to_fp8_scalar(v, e) for v in vals)
    assert got == want
    for x in vals:
        _check_encode(x, e)


# ---------------------------------------------------------------------------
# Cross-check against ml_dtypes (finite encodings identical for E3, E4, E5)
# ---------------------------------------------------------------------------

_ML_DTYPES = [
    (DType.E3, ml_dtypes.float8_e3m4),
    (DType.E4, ml_dtypes.float8_e4m3),
    (DType.E5, ml_dtypes.float8_e5m2),
]


def _ml_encode(xs: np.ndarray, ml_type) -> bytes:
    return np.asarray(xs, dtype=np.float32).astype(ml_type).view(np.uint8).tobytes()


@pytest.mark.parametrize("dtype,ml_type", _ML_DTYPES, ids=lambda p: getattr(p, "name", str(p)))
def test_ml_dtypes_cross_check(dtype, ml_type):
    e = int(dtype)
    max_mag = float(_TABLES[e][-1][0])

    mids = np.array(_midpoint_inputs(e), dtype=np.float32)
    assert fp32_to_fp8_bytes(mids, dtype) == _ml_encode(mids, ml_type)

    rng = np.random.default_rng(0xE0001)
    n = 100_000
    # Log-uniform magnitudes spanning zero-rounding through max_finite, so the
    # subnormal and every normal binade are sampled.
    _, bias = _fields(e)
    lo_exp = -bias - (7 - e) - 3
    mags = np.exp2(rng.uniform(lo_exp, math.log2(max_mag), n))
    signs = rng.choice([-1.0, 1.0], n)
    xs = np.clip((signs * mags).astype(np.float32), -max_mag, max_mag)
    got = fp32_to_fp8_bytes(xs, dtype)
    want = _ml_encode(xs, ml_type)
    if got != want:
        bad = next(i for i in range(n) if got[i] != want[i])
        pytest.fail(f"x={xs[bad]!r}: got {got[bad]:#04x}, ml_dtypes {want[bad]:#04x}")

    codes = [c for c in range(256) if not ref_is_nan(c, e)]
    ours = fp8_bytes_to_fp32(bytes(codes), dtype)
    theirs = np.array(codes, dtype=np.uint8).view(ml_type).astype(np.float32)
    np.testing.assert_array_equal(ours.view(np.uint32), theirs.view(np.uint32))
