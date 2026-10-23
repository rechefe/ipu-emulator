"""The two run modes' element formats and their derived sizes (ADR 0007)."""

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


def test_exactly_two_formats():
    assert [str(f) for f in ElementFormat] == ["fp32", "nx8"]
    assert ElementFormat("nx8") is ElementFormat.NX8
    assert list(AccumulatorType) == [AccumulatorType.FP32]


def test_every_format_listed_once():
    assert list(FORMATS) == list(ElementFormat)
    assert all(info.format is fmt for fmt, info in FORMATS.items())


def test_tags():
    assert FORMATS[ElementFormat.NX8].tag is Tag.HARDWARE
    assert FORMATS[ElementFormat.FP32].tag is Tag.EMULATOR


def test_sizes():
    nx8, fp32 = FORMATS[ElementFormat.NX8], FORMATS[ElementFormat.FP32]
    assert (nx8.element_bytes, nx8.row_bytes) == (1, 128)
    assert (fp32.element_bytes, fp32.row_bytes) == (4, 512)
    assert nx8.has_metadata and nx8.metadata_bits == 16
    assert not fp32.has_metadata and fp32.metadata_bits == 0
    assert all(info.accumulator_bytes == 4 for info in FORMATS.values())


def test_register_layout():
    assert register_layout("r", ElementFormat.FP32) == (128, 4)
    assert register_layout("r", ElementFormat.NX8) == (128, 1)
    assert register_layout("r_cyclic", ElementFormat.NX8) == (512, 1)
    assert register_layout("r_cyclic", ElementFormat.FP32) == (512, 4)
    for fmt in ElementFormat:
        assert register_layout("r_acc", fmt) == (LANES, 4)
        assert register_layout("mult_res", fmt) == (LANES, 4)
        assert register_layout("r_mask", fmt) == (128, 1)
    with pytest.raises(KeyError):
        register_layout("lr", ElementFormat.NX8)
    with pytest.raises(KeyError):
        register_layout("no_such_register", ElementFormat.NX8)


def test_addressable_rows():
    assert addressable_rows(ElementFormat.FP32) == XMEM_BYTES // 512
    assert addressable_rows(ElementFormat.NX8) == XMEM_BYTES // 128


def _with(fmt: ElementFormat, **changes: object) -> dict[ElementFormat, FormatInfo]:
    formats = dict(FORMATS)
    formats[fmt] = dataclasses.replace(formats[fmt], **changes)  # type: ignore[arg-type]
    return formats


@pytest.mark.parametrize(
    ("fmt", "changes", "message"),
    [
        (ElementFormat.FP32, {"tag": Tag.ASSEMBLER}, "HARDWARE or EMULATOR"),
        (ElementFormat.NX8, {"element_bits": 7}, "whole bytes"),
        (ElementFormat.NX8, {"format": ElementFormat.FP32}, "keyed under"),
    ],
)
def test_schema_rules(fmt, changes, message):
    with pytest.raises(SpecError, match=message):
        validate_formats(_with(fmt, **changes))


def test_missing_format_rejected():
    formats = dict(FORMATS)
    del formats[ElementFormat.NX8]
    with pytest.raises(SpecError, match="missing"):
        validate_formats(formats)


def test_records_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FORMATS[ElementFormat.NX8].element_bits = 16  # type: ignore[misc]
    with pytest.raises(TypeError):
        FORMATS[ElementFormat.NX8] = FORMATS[ElementFormat.FP32]  # type: ignore[index]
