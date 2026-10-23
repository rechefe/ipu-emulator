"""The AaQ LUT layout and the lut_image builder (ADR 0007, provisional entries)."""

import math

import pytest

from ipu_isa.errors import IsaValueError
from ipu_isa.lut import (
    SEGMENT_BYTES,
    TABLE_BITS,
    decode_entry,
    decode_metadata,
    encode_entry,
    entry_points,
    load_segment,
    lut_image,
    segment_payload,
    table_entries,
)


def _load(rows):
    table, meta = 0, None
    for addr, row in enumerate(rows):
        table, new_meta = load_segment(table, addr, segment_payload(row))
        meta = new_meta if new_meta is not None else meta
    return table, meta


def test_five_rows_fill_the_table_and_metadata():
    rows = lut_image(lambda x: 1 / (1 + math.exp(-x)), range_mode=2, range_exp=3, pos_c=1.0)
    assert len(rows) == 5
    assert all(len(r) == SEGMENT_BYTES for r in rows)
    table, meta = _load(rows)
    assert table < 1 << TABLE_BITS
    fields = decode_metadata(meta)
    assert fields.max_exp_num == 3 + 127
    assert fields.range_mode == 2
    assert decode_entry(fields.pos_c) == 1.0
    points = entry_points(2, 3)
    for code, x in zip(table_entries(table), points):
        assert decode_entry(code) == pytest.approx(1 / (1 + math.exp(-x)), rel=2**-8)


def test_version3_entry_points_match_the_spec_example():
    points = entry_points(2, 1)  # range -2..2, interval 4/256
    assert points[0] == -2 + 4 / 256
    assert points[127] == 0.0 and math.copysign(1, points[127]) == -1
    assert points[128] == 0.0 and math.copysign(1, points[128]) == 1
    assert points[-1] == 2 - 4 / 256


def test_entry_round_trip():
    assert decode_entry(encode_entry(1.5)) == 1.5
    assert decode_entry(encode_entry(-0.25)) == -0.25


def test_segment_needs_128_bytes_and_a_valid_address():
    with pytest.raises(IsaValueError):
        segment_payload(bytes(64))
    with pytest.raises(IsaValueError):
        load_segment(0, 5, 0)
    with pytest.raises(IsaValueError):
        entry_points(3, 0)
