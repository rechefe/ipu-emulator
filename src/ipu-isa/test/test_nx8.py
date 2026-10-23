"""NX8 quantizer and expansion: exact bytes from the spec's rules (ADR 0007)."""

import math

import pytest

from ipu_isa.errors import IsaValueError
from ipu_isa.nx8 import (
    Nx8Format,
    choose_dynamic_exponent,
    decode_element,
    decode_row,
    dynamic_scores,
    encode_row,
    row_scale,
)

E4M3 = Nx8Format(sign=1, exp_format=4, num_bits=8)
E2M5 = Nx8Format(sign=1, exp_format=2, num_bits=8)


def test_worked_row_direct_branch_edge1_and_zero():
    # 3.2 sets the scale (raw exponent 128); -1.99 carries one binade up (edge 1);
    # 0 falls into the clamp branch with M = 0.
    row = encode_row([3.2, 0.4, -1.99, 0.0], E4M3)
    assert row.scale == 128
    assert list(row.elements[:4]) == [0x05, 0x1D, 0x80, 0x78]
    assert row.elements[4:] == bytes(124)
    assert not row.saturated
    assert decode_row(row.elements[:4], E4M3, row.scale) == [3.25, 0.40625, -2.0, 0.0]


def test_edge1_at_the_scale_saturates():
    row = encode_row([3.99], E4M3)
    assert row.elements[0] == 0x07
    assert decode_element(0x07, E4M3, row.scale) == 3.75
    assert row.saturated


def test_edge2_carries_into_the_smallest_direct_exponent():
    row = encode_row([3.2, 0.4975], E2M5)
    assert list(row.elements[:2]) == [0x13, 0x40]
    assert decode_element(0x40, E2M5, row.scale) == 0.5


def test_clamp_branch_keeps_magnitude():
    row = encode_row([3.2, 0.001], E4M3)
    assert row.elements[1] == 0x58
    assert decode_element(0x58, E4M3, row.scale) == 2.0**-10


def test_ties_round_half_up():
    # 1.0625 is exactly halfway between 1.0 and 1.125 at 3 mantissa bits.
    row = encode_row([1.0625], E4M3)
    assert decode_element(row.elements[0], E4M3, row.scale) == 1.125


def test_inactive_elements_are_zero_and_leave_the_scale():
    row = encode_row([100.0, 1.5], E4M3, active=[False, True])
    assert row.scale == 127
    assert row.elements[0] == 0
    assert decode_element(row.elements[1], E4M3, row.scale) == 1.5


def test_infinity_saturates_and_leaves_the_scale():
    row = encode_row([math.inf, -math.inf, 1.0], E4M3)
    assert row.scale == 127
    assert list(row.elements[:2]) == [0x07, 0x87]
    assert row.saturated


def test_nan_raises():
    with pytest.raises(IsaValueError, match="NaN"):
        encode_row([math.nan], E4M3)


def test_negative_into_unsigned_is_zero_and_saturates():
    unsigned = Nx8Format(sign=0, exp_format=4, num_bits=8)
    row = encode_row([-1.0, 1.0], unsigned)
    assert decode_element(row.elements[0], unsigned, row.scale) == 0.0
    assert row.saturated
    assert not encode_row([-0.0, 1.0], unsigned).saturated


def test_narrow_width_pads_low_bits():
    e2m3 = Nx8Format(sign=1, exp_format=2, num_bits=6)
    row = encode_row([1.5], e2m3)
    assert row.elements[0] & 0b11 == 0
    assert decode_element(row.elements[0], e2m3, row.scale) == 1.5


def test_dynamic_worked_example_scores():
    # ARCH-stage-aaq-str.md §4.3.2.3: scale 0 (unbiased), W = 6.
    values = [1.83, 1.21, 1.57, 0.94, 0.67, 0.41, 0.2345, 0.1780, 0.0920, 0.0555]
    assert dynamic_scores(values, num_bits=6) == (5, 2)


def test_dynamic_row_records_its_split():
    fmt = Nx8Format(sign=1, exp_format=0, num_bits=8, dynamic=True)
    values = [1.0, 0.24, 0.12, 0.06, 0.03]  # mostly far below the scale: E2M5 wins
    row = encode_row(values, fmt)
    assert row.dynamic_exponent == choose_dynamic_exponent(values)
    for code, value in zip(row.elements[: len(values)], values, strict=True):
        decoded = decode_element(code, fmt, row.scale, row.dynamic_exponent)
        assert abs(decoded - value) <= abs(value) * 2**-4


@pytest.mark.parametrize("fe", range(1, 8))
def test_every_code_round_trips(fe):
    fmt = Nx8Format(sign=1, exp_format=fe, num_bits=8)
    scale = 140
    for code in range(256):
        value = decode_element(code, fmt, scale)
        if value == 0.0:
            continue
        row = encode_row([math.ldexp(1.0, scale - 127), value], fmt)  # pins the scale
        assert decode_element(row.elements[1], fmt, scale) == value


def test_row_scale_uses_active_finite_elements():
    assert row_scale([8.0, 1.0], active=[False, True]) == 127
    assert row_scale([]) == 0


@pytest.mark.parametrize(
    "fields",
    [
        dict(sign=2, exp_format=4, num_bits=8),
        dict(sign=1, exp_format=0, num_bits=8),
        dict(sign=1, exp_format=8, num_bits=8),
        dict(sign=1, exp_format=4, num_bits=0),
        dict(sign=1, exp_format=7, num_bits=7),
        dict(sign=0, exp_format=0, num_bits=8, dynamic=True),
        dict(sign=1, exp_format=0, num_bits=6, dynamic=True),
    ],
)
def test_invalid_formats_raise(fields):
    with pytest.raises(IsaValueError):
        Nx8Format(**fields)
