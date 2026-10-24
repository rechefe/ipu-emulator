"""Display helpers for the debugger's panes.

Field widths, the 16-bit character/colour cell layout, the visible alphabet
and the XMEM tokenizer live here, apart from curses, so every pane lays values
out the same way without reaching into another pane's module.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass

from ipu_debug.session import XMEM_FORMAT_ITEM_SIZE_BYTES
from ipu_debug.stores import metadata_text
from ipu_emu.xmem import XMEM_WIDTH_BYTES

__all__ = [
    "CELL16_BACKGROUND_SHIFT",
    "CELL16_CHARACTER_MASK",
    "CELL16_CHARACTERS_PER_GROUP",
    "CELL16_COLOR_MASK",
    "CELL16_FOREGROUND_SHIFT",
    "F32_DISPLAY_WIDTH",
    "F32_SIGNIFICANT_DIGITS",
    "HEX_BYTES_PER_GROUP",
    "INT8_DECIMAL_WIDTH",
    "U8_DECIMAL_WIDTH",
    "U32_DECIMAL_WIDTH",
    "VISIBLE_ALPHABET",
    "XMEM_VALUES_PER_LINE",
    "XmemDisplayLine",
    "XmemDisplayToken",
    "decode_16bit_cell",
    "tokenize_xmem",
]

XMEM_VALUES_PER_LINE = {
    "hex": 16,
    "int8": 16,
    "u8": 16,
    "cell16": 32,
    "u32": 4,
    "f32": 4,
}

HEX_BYTES_PER_GROUP = 4
CELL16_CHARACTERS_PER_GROUP = 2
INT8_DECIMAL_WIDTH = 4
U8_DECIMAL_WIDTH = 3
U32_DECIMAL_WIDTH = 10
F32_DISPLAY_WIDTH = 15
F32_SIGNIFICANT_DIGITS = 9

CELL16_CHARACTER_MASK = 0xFF
CELL16_COLOR_MASK = 0xF
CELL16_FOREGROUND_SHIFT = 8
CELL16_BACKGROUND_SHIFT = 12

# A control-free character for every possible 8-bit cell value. The ASCII and
# Latin-1 ranges provide 188 entries; Latin Extended-A supplies the remaining
# 68 so all indices 0..255 are valid.
_VISIBLE_ASCII = [chr(i) for i in range(33, 127)]
_VISIBLE_LATIN1 = [chr(i) for i in range(161, 256) if i != 173]
_VISIBLE_EXTENDED = [chr(i) for i in range(0x100, 0x144)]
VISIBLE_ALPHABET = _VISIBLE_ASCII + _VISIBLE_LATIN1 + _VISIBLE_EXTENDED
assert len(VISIBLE_ALPHABET) == 256


@dataclass(frozen=True)
class XmemDisplayToken:
    """One formatted XMEM value plus its source-byte span."""

    text: str
    leading: str
    raw_start: int
    raw_size: int
    fg: int | None = None
    bg: int | None = None


@dataclass(frozen=True)
class XmemDisplayLine:
    """One address or row-marker line in a formatted XMEM display."""

    prefix: str
    tokens: tuple[XmemDisplayToken, ...] = ()


def decode_16bit_cell(char_code: int, fg_4bit: int, bg_4bit: int) -> tuple[str, int, int]:
    """Return the visible character and normalized 4-bit colors."""
    return (
        VISIBLE_ALPHABET[char_code & CELL16_CHARACTER_MASK],
        fg_4bit & CELL16_COLOR_MASK,
        bg_4bit & CELL16_COLOR_MASK,
    )


def _format_item(data: bytes | bytearray, raw_start: int, fmt: str) -> str:
    if fmt == "hex":
        return f"{data[raw_start]:02x}"
    if fmt == "int8":
        return f"{struct.unpack_from('<b', data, raw_start)[0]:>{INT8_DECIMAL_WIDTH}d}"
    if fmt == "u8":
        return f"{data[raw_start]:0{U8_DECIMAL_WIDTH}d}"
    if fmt == "u32":
        return f"{struct.unpack_from('<I', data, raw_start)[0]:0{U32_DECIMAL_WIDTH}d}"
    value = struct.unpack_from("<f", data, raw_start)[0]
    return f"{value:>{F32_DISPLAY_WIDTH}.{F32_SIGNIFICANT_DIGITS}g}"


def _tokenize_line(
    data: bytes | bytearray, start: int, count: int, fmt: str, row_size: int
) -> tuple[XmemDisplayToken, ...]:
    item_size = XMEM_FORMAT_ITEM_SIZE_BYTES[fmt]
    tokens: list[XmemDisplayToken] = []
    for i in range(count):
        raw_start = (start + i) * item_size
        leading = " "
        fg: int | None = None
        bg: int | None = None
        if fmt == "cell16":
            cell = struct.unpack_from("<H", data, raw_start)[0]
            fg = (cell >> CELL16_FOREGROUND_SHIFT) & CELL16_COLOR_MASK
            bg = (cell >> CELL16_BACKGROUND_SHIFT) & CELL16_COLOR_MASK
            text = VISIBLE_ALPHABET[cell & CELL16_CHARACTER_MASK]
            leading = " " if i == 0 or i % CELL16_CHARACTERS_PER_GROUP == 0 else ""
        else:
            if fmt == "hex" and row_size > XMEM_WIDTH_BYTES and i > 0:
                leading = "  " if i % HEX_BYTES_PER_GROUP == 0 else " "
            text = _format_item(data, raw_start, fmt)
        tokens.append(XmemDisplayToken(text, leading, raw_start, item_size, fg, bg))
    return tuple(tokens)


def tokenize_xmem(
    data: bytes | bytearray,
    byte_address: int,
    fmt: str,
    row_size: int,
    row_metadata: Mapping[int, int] | None = None,
) -> list[XmemDisplayLine]:
    """Tokenize validated XMEM data for curses rendering.

    Each row's marker line names the row's metadata word when *row_metadata*
    (row -> word, NX8 mode) holds it.
    """
    values_per_line = XMEM_VALUES_PER_LINE[fmt]
    item_size = XMEM_FORMAT_ITEM_SIZE_BYTES[fmt]
    lines: list[XmemDisplayLine] = []
    total_items = len(data) // item_size
    start = 0

    while start < total_items:
        address = byte_address + start * item_size
        row = address // row_size
        row_address = row * row_size
        if start == 0 or address == row_address:
            marker = f"--- XMEM row {row}"
            if row_metadata is not None and row in row_metadata:
                marker += f" [{metadata_text(row_metadata[row])}]"
            marker += f" starts at byte address 0x{row_address:08x}"
            if address != row_address:
                marker += f"; display starts at 0x{address:08x}"
            lines.append(XmemDisplayLine(marker + " ---"))

        items_left_in_row = (row_address + row_size - address) // item_size
        count = min(values_per_line, total_items - start, items_left_in_row)
        tokens = _tokenize_line(data, start, count, fmt, row_size)
        lines.append(XmemDisplayLine(f"  0x{address:08x}:", tokens))
        start += count

    return lines
