"""Element formats and their derived sizes (E03-13)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from ipu_isa.formats import (
    FORMATS,
    AccumulatorType,
    ElementFormat,
    FormatInfo,
    addressable_rows,
    decode_element,
    element_values,
    encode_element,
    register_layout,
    validate_formats,
)
from ipu_isa.geometry import LANES, XMEM_BYTES
from ipu_isa.schema import SpecError
from ipu_isa.tags import Tag

EIGHT_BIT_FLOATS = [ElementFormat(f"e{x}") for x in range(1, 8)]


def test_names_are_lower_case():
    assert [str(f) for f in ElementFormat] == [
        "int8",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "fp32",
        "int32",
    ]
    assert ElementFormat("e4") is ElementFormat.E4


def test_every_format_listed_once():
    assert list(FORMATS) == list(ElementFormat)
    assert all(info.format is fmt for fmt, info in FORMATS.items())


def test_tags():
    hardware = {f for f, info in FORMATS.items() if info.tag is Tag.HARDWARE}
    emulator = {f for f, info in FORMATS.items() if info.tag is Tag.EMULATOR}
    assert hardware == {ElementFormat.INT8, *EIGHT_BIT_FLOATS}
    assert emulator == {ElementFormat.FP32, ElementFormat.INT32}


def test_sizes():
    assert FORMATS[ElementFormat.INT8].row_bytes == 128
    assert FORMATS[ElementFormat.FP32].row_bytes == 512
    for fmt in (ElementFormat.INT8, *EIGHT_BIT_FLOATS):
        assert FORMATS[fmt].element_bits == 8
        assert FORMATS[fmt].element_bytes == 1
        assert FORMATS[fmt].row_bytes == LANES
    for fmt in (ElementFormat.FP32, ElementFormat.INT32):
        assert FORMATS[fmt].element_bytes == 4
        assert FORMATS[fmt].row_bytes == 4 * LANES


def test_bit_layouts():
    for x, fmt in enumerate(EIGHT_BIT_FLOATS, start=1):
        info = FORMATS[fmt]
        assert info.is_float
        assert (info.exp_bits, info.man_bits) == (x, 7 - x)
    fp32 = FORMATS[ElementFormat.FP32]
    assert (fp32.exp_bits, fp32.man_bits) == (8, 23)
    for fmt in (ElementFormat.INT8, ElementFormat.INT32):
        info = FORMATS[fmt]
        assert not info.is_float
        assert (info.exp_bits, info.man_bits) == (0, 0)


def test_accumulator_rule():
    assert FORMATS[ElementFormat.INT8].accumulator is AccumulatorType.INT32
    assert FORMATS[ElementFormat.INT32].accumulator is AccumulatorType.INT32
    for fmt in (*EIGHT_BIT_FLOATS, ElementFormat.FP32):
        assert FORMATS[fmt].accumulator is AccumulatorType.FP32
    assert all(info.accumulator_bytes == 4 for info in FORMATS.values())


def test_register_layout():
    assert register_layout("r", ElementFormat.FP32) == (128, 4)
    assert register_layout("r", ElementFormat.INT8) == (128, 1)
    assert register_layout("r_cyclic", ElementFormat.E4) == (512, 1)
    assert register_layout("r_cyclic", ElementFormat.INT32) == (512, 4)
    for fmt in ElementFormat:
        assert register_layout("r_acc", fmt) == (LANES, 4)
        assert register_layout("mult_res", fmt) == (LANES, 4)
        assert register_layout("r_mask", fmt) == (128, 1)
    with pytest.raises(KeyError):
        register_layout("lr", ElementFormat.INT8)
    with pytest.raises(KeyError):
        register_layout("no_such_register", ElementFormat.INT8)


def test_addressable_rows():
    assert addressable_rows(ElementFormat.FP32) == XMEM_BYTES // 512
    assert addressable_rows(ElementFormat.INT8) == XMEM_BYTES // 128


def _with(fmt: ElementFormat, **changes: object) -> dict[ElementFormat, FormatInfo]:
    formats = dict(FORMATS)
    formats[fmt] = dataclasses.replace(formats[fmt], **changes)  # type: ignore[arg-type]
    return formats


@pytest.mark.parametrize(
    ("fmt", "changes", "message"),
    [
        (ElementFormat.FP32, {"tag": Tag.ASSEMBLER}, "HARDWARE or EMULATOR"),
        (ElementFormat.E4, {"man_bits": 4}, "mantissa"),
        (ElementFormat.E4, {"accumulator": AccumulatorType.INT32}, "accumulator"),
        (ElementFormat.INT8, {"accumulator": AccumulatorType.FP32}, "accumulator"),
        (ElementFormat.INT8, {"exp_bits": 1}, "integer format"),
        (ElementFormat.FP32, {"exp_bits": 11, "man_bits": 20}, "IEEE"),
        (ElementFormat.E1, {"element_bits": 7, "man_bits": 5}, "whole number"),
        (ElementFormat.E2, {"format": ElementFormat.E3}, "keyed under"),
    ],
)
def test_schema_rules(fmt, changes, message):
    with pytest.raises(SpecError, match=message):
        validate_formats(_with(fmt, **changes))


def test_missing_format_rejected():
    formats = dict(FORMATS)
    del formats[ElementFormat.INT32]
    with pytest.raises(SpecError, match="missing"):
        validate_formats(formats)


def test_records_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FORMATS[ElementFormat.INT8].element_bits = 16  # type: ignore[misc]
    with pytest.raises(TypeError):
        FORMATS[ElementFormat.INT8] = FORMATS[ElementFormat.E4]  # type: ignore[index]


# --- the 8-bit element codec (E04-13) ----------------------------------------


@pytest.mark.parametrize(
    ("code", "fmt", "value"),
    [
        (0x38, ElementFormat.E4, 1.0),
        (0x40, ElementFormat.E4, 2.0),
        (0x77, ElementFormat.E4, 240.0),  # largest finite: exponent 14, mantissa 7
        (0x01, ElementFormat.E4, 2.0**-9),  # smallest subnormal
        (0xB8, ElementFormat.E4, -1.0),
        (0x3C, ElementFormat.E5, 1.0),
        (0x01, ElementFormat.E1, 2.0**-5),  # E1: bias 0, all subnormal
        (0x3E, ElementFormat.E7, 2.0**-1),  # E7: powers of two only
        (0x7F, ElementFormat.INT8, 127.0),
        (0xFF, ElementFormat.INT8, -1.0),
        (0x80, ElementFormat.INT8, -128.0),
    ],
)
def test_decode_element_known_codes(code, fmt, value):
    assert decode_element(code, fmt) == value


def test_decode_element_nan_and_negative_zero():
    assert math.isnan(decode_element(0x7F, ElementFormat.E4))
    assert math.copysign(1.0, decode_element(0x80, ElementFormat.E4)) == -1.0


@pytest.mark.parametrize("fmt", [f for f, i in FORMATS.items() if i.element_bits == 8], ids=str)
def test_encode_inverts_decode_on_every_code(fmt):
    values = element_values(fmt)
    assert len(values) == 256
    for code, value in enumerate(values):
        if math.isnan(value):
            continue
        assert encode_element(value, fmt) == code


@pytest.mark.parametrize(
    ("value", "fmt", "code"),
    [
        (1.0625, ElementFormat.E4, 0x38),  # tie between 1.0 and 1.125: even code
        (1.1875, ElementFormat.E4, 0x3A),  # tie between 1.125 and 1.25: even code
        (1e6, ElementFormat.E4, 0x77),  # saturates
        (-math.inf, ElementFormat.E4, 0xF7),
        (math.nan, ElementFormat.E4, 0x79),  # canonical NaN
        (-0.0, ElementFormat.E4, 0x80),
        (127.6, ElementFormat.INT8, 0x7F),
        (-200.0, ElementFormat.INT8, 0x80),
        (2.5, ElementFormat.INT8, 0x02),
        (-1.0, ElementFormat.INT8, 0xFF),
    ],
)
def test_encode_element_rounds_and_saturates(value, fmt, code):
    assert encode_element(value, fmt) == code


def test_element_codec_rejects_other_formats():
    for fmt in (ElementFormat.FP32, ElementFormat.INT32):
        with pytest.raises(ValueError, match="not an 8-bit format"):
            decode_element(0, fmt)
        with pytest.raises(ValueError, match="not an 8-bit format"):
            encode_element(1.0, fmt)
    with pytest.raises(ValueError, match="not a byte"):
        decode_element(256, ElementFormat.E4)
    with pytest.raises(ValueError, match="NaN"):
        encode_element(math.nan, ElementFormat.INT8)
