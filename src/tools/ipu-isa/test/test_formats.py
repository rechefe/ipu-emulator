"""Element formats and their derived sizes (E03-13)."""

from __future__ import annotations

import dataclasses

import pytest

from ipu_isa.formats import (
    FORMATS,
    AccumulatorType,
    ElementFormat,
    FormatInfo,
    addressable_rows,
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
