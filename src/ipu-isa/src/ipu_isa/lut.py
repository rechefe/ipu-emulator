"""The AaQ activation LUT: its bit layout and a builder for test images (ADR 0007).

The LUT is a flat 4,352-bit table (256 entries of 17 bits, entry ``i`` at table
bits ``17*i`` up) plus a 46-bit metadata block. Five ``LD.LUT`` loads fill it,
one 1024-bit segment each: segments 0-3 are table bits ``[1024*k +: 1024]``;
segment 4 holds table bits ``[4351:4096]`` in payload bits ``[255:0]`` and the
metadata in ``[301:256]`` (``ARCH-stage-aaq-str.md`` §4.2). A segment is the
first 128 bytes of an XMEM row, little-endian: payload bit ``8*b + j`` is bit
``j`` of byte ``b``.

Provisional (OQ-38): the specs do not define the 17-bit entry format nor how a
value between entry points is computed. :func:`encode_entry` uses the top 17
bits of the FP32 pattern (sign, exponent, 8 mantissa bits), and ``max_exp_num``
holds the biased exponent of the range, ``range_exp + 127`` (§4.2.2), while the
entry points use ``2^range_exp`` (§4.2.3). The emulator stores the bits exactly
and evaluates activations with exact math until the format is published.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from ipu_isa.cr_halves import Field
from ipu_isa.errors import IsaValueError

ENTRIES: Final = 256
ENTRY_BITS: Final = 17
TABLE_BITS: Final = ENTRIES * ENTRY_BITS  # 4352
SEGMENT_BITS: Final = 1024
SEGMENT_BYTES: Final = SEGMENT_BITS // 8
SEGMENTS: Final = 5
METADATA_LSB: Final = 256  # in segment 4's payload
_SCALE_BIAS: Final = 127

#: The 46-bit metadata block, bits local to the block (§4.2.2).
METADATA_LAYOUT: Final = {
    "max_exp_num": Field(0, 8),
    "neg_c": Field(8, 17),
    "pos_c": Field(25, 17),
    "range_mode": Field(42, 2),
    "neg_mode": Field(44, 1),
    "pos_mode": Field(45, 1),
}
METADATA_BITS: Final = 46


@dataclass(frozen=True)
class OutOfRange:
    """What the LUT returns outside its range: ``x`` (mode 0) or the constant (mode 1)."""

    neg_c: float = 0.0
    pos_c: float = 0.0
    neg_mode: int = 0
    pos_mode: int = 0


@dataclass(frozen=True)
class LutMetadata:
    """The decoded metadata block."""

    max_exp_num: int
    neg_c: int
    pos_c: int
    range_mode: int  # 0, 1, 2 = versions 1, 2, 3; 3 is undefined (OQ-38)
    neg_mode: int
    pos_mode: int


def encode_entry(value: float) -> int:
    """Provisional 17-bit entry: the top 17 bits of the FP32 pattern, rounded (OQ-38)."""
    bits = int(struct.unpack(">I", struct.pack(">f", value))[0])
    return min((bits + (1 << 14)) >> 15, (1 << ENTRY_BITS) - 1)


def decode_entry(code: int) -> float:
    """The value of a provisional 17-bit entry."""
    return float(struct.unpack(">f", struct.pack(">I", (code & ((1 << ENTRY_BITS) - 1)) << 15))[0])


def entry_points(range_mode: int, range_exp: int) -> list[float]:
    """The x of each entry, in entry order (§4.2.3); version 3 runs from the most negative."""
    span = 2.0**range_exp
    step = span / ENTRIES
    if range_mode == 0:
        return [i * step for i in range(ENTRIES)]
    if range_mode == 1:
        return [-i * step for i in range(ENTRIES)]
    if range_mode == 2:
        half = ENTRIES // 2
        negative = [-(span - 2 * step) + 2 * step * i for i in range(half - 1)] + [-0.0]
        positive = [2 * step * i for i in range(half)]
        return negative + positive
    raise IsaValueError(f"range_mode {range_mode} is undefined (OQ-38)")


def _set_field(bits: int, field: Field, value: int, offset: int = 0) -> int:
    return bits | (field.put(value) << offset)


def lut_image(
    fn: Callable[[float], float],
    range_mode: int,
    range_exp: int,
    out_of_range: OutOfRange | None = None,
) -> list[bytes]:
    """The five 128-byte rows that load *fn* into the LUT with ``LD.LUT 0``-``4``.

    A test or kernel places the rows in XMEM (in either mode, ``LD.LUT`` reads a
    row's first 128 bytes) and declares *fn* as the loaded function, which the
    emulator evaluates exactly until the entry format is published (OQ-38).
    """
    out_of_range = out_of_range or OutOfRange()
    table = 0
    for i, x in enumerate(entry_points(range_mode, range_exp)):
        table |= encode_entry(fn(x)) << (ENTRY_BITS * i)
    meta = 0
    for name, value in (
        ("max_exp_num", range_exp + _SCALE_BIAS),
        ("neg_c", encode_entry(out_of_range.neg_c)),
        ("pos_c", encode_entry(out_of_range.pos_c)),
        ("range_mode", range_mode),
        ("neg_mode", out_of_range.neg_mode),
        ("pos_mode", out_of_range.pos_mode),
    ):
        meta = _set_field(meta, METADATA_LAYOUT[name], value)
    segments = [(table >> (SEGMENT_BITS * k)) & ((1 << SEGMENT_BITS) - 1) for k in range(4)]
    segments.append((table >> (4 * SEGMENT_BITS)) | (meta << METADATA_LSB))
    return [s.to_bytes(SEGMENT_BYTES, "little") for s in segments]


def segment_payload(row: bytes) -> int:
    """The 1024-bit payload ``LD.LUT`` takes from a row: its first 128 bytes."""
    if len(row) < SEGMENT_BYTES:
        raise IsaValueError(f"a LUT segment needs {SEGMENT_BYTES} bytes, got {len(row)}")
    return int.from_bytes(row[:SEGMENT_BYTES], "little")


def load_segment(table: int, lut_addr: int, payload: int) -> tuple[int, int | None]:
    """``(table, metadata or None)`` after loading *payload* into segment *lut_addr*."""
    if not 0 <= lut_addr < SEGMENTS:
        raise IsaValueError(f"lut_addr {lut_addr} is not 0-4")
    if lut_addr < 4:
        mask = ((1 << SEGMENT_BITS) - 1) << (SEGMENT_BITS * lut_addr)
        return (table & ~mask) | (payload << (SEGMENT_BITS * lut_addr)), None
    tail = TABLE_BITS - 4 * SEGMENT_BITS
    mask = ((1 << tail) - 1) << (4 * SEGMENT_BITS)
    table = (table & ~mask) | ((payload & ((1 << tail) - 1)) << (4 * SEGMENT_BITS))
    return table, (payload >> METADATA_LSB) & ((1 << METADATA_BITS) - 1)


def decode_metadata(meta: int) -> LutMetadata:
    """The fields of a metadata block."""
    return LutMetadata(**{name: f.get(meta) for name, f in METADATA_LAYOUT.items()})


def table_entries(table: int) -> list[int]:
    """The 256 entry codes of a table bitstream."""
    mask = (1 << ENTRY_BITS) - 1
    return [(table >> (ENTRY_BITS * i)) & mask for i in range(ENTRIES)]
