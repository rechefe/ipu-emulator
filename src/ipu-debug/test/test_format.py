"""The XMEM display tokens of ``ipu_debug.format`` on real states.

Each test reads an XMEM window through ``DebugSession`` (the request text the
XMEM tab editor takes) and tokenises it the way the XMEM pane does, so the
layout is checked without a screen.
"""

from __future__ import annotations

import struct

import pytest
from emu_support import ElementFormat, new_state

from ipu_debug.format import VISIBLE_ALPHABET, decode_16bit_cell, tokenize_xmem
from ipu_debug.session import DebugError, DebugSession
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import XMEM_SIZE_BYTES


def show(state: IpuState, text: str) -> str:
    """The XMEM window *text* names, one display line per text line."""
    session = DebugSession(state)
    window = session.read_xmem(session.parse_xmem(text))
    lines = tokenize_xmem(window.data, window.byte_address, window.request.fmt, window.row_size)
    return "\n".join(
        line.prefix + "".join(token.leading + token.text for token in line.tokens) for line in lines
    )


def value_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("  0x")]


class TestXmemLayout:
    def test_hex_bytes_at_a_byte_address(self):
        state = IpuState()
        state.xmem.write_address(0x100, bytes([0x12, 0x34, 0xAB, 0xCD]))
        assert "  0x00000100: 12 34 ab cd" in show(state, "byte 0x100 0 4 hex")

    def test_fp32_hex_groups_every_four_bytes(self):
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(0, bytes(range(8)))
        assert "00 01 02 03  04 05 06 07" in show(state, "row 0 0 8 hex")

    def test_int8_and_u8_formats(self):
        state = IpuState()
        state.xmem.write_address(0, bytes([0, 127, 128, 255]))
        assert f"{0:>4d} {127:>4d} {-128:>4d} {-1:>4d}" in show(state, "byte 0 0 4 int8")
        assert "000 127 128 255" in show(state, "byte 0 0 4 u8")

    def test_cell16_shows_the_character_and_keeps_its_colours(self):
        state = IpuState()
        state.xmem.write_address(0, struct.pack("<H", 200 | (14 << 8) | (4 << 12)))
        session = DebugSession(state)
        window = session.read_xmem(session.parse_xmem("byte 0 0 1 cell16"))
        lines = tokenize_xmem(window.data, 0, "cell16", window.row_size)
        assert lines[0].prefix == "--- XMEM row 0 starts at byte address 0x00000000 ---"
        (token,) = lines[1].tokens
        assert (token.text, token.fg, token.bg) == (VISIBLE_ALPHABET[200], 14, 4)

    def test_cell16_separates_every_two_characters(self):
        state = IpuState()
        state.xmem.write_address(
            0, struct.pack("<4H", *(c | (1 << 8) | (2 << 12) for c in range(4)))
        )
        expected = " " + "".join(VISIBLE_ALPHABET[:2]) + " " + "".join(VISIBLE_ALPHABET[2:4])
        assert value_lines(show(state, "byte 0 0 4 cell16")) == ["  0x00000000:" + expected]

    def test_cell16_shows_32_characters_per_line(self):
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(0, struct.pack("<33H", *range(33)))
        lines = value_lines(show(state, "row 0 0 33 cell16"))
        assert len(lines) == 2
        assert lines[0].startswith("  0x00000000:")
        assert lines[1].startswith("  0x00000040:")

    def test_f32_keeps_distinct_float32_values_apart(self):
        state = IpuState()
        state.xmem.write_address(0, struct.pack("<2I", 0x3F800001, 0x3F800002))
        shown = show(state, "byte 0 0 2 f32")
        assert "1.00000012" in shown
        assert "1.00000024" in shown

    def test_each_xmem_row_boundary_is_marked(self):
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(508, bytes(range(8)))
        shown = show(state, "byte 508 0 8 u8")
        assert (
            "XMEM row 0 starts at byte address 0x00000000; display starts at 0x000001fc"
        ) in shown
        assert "0x000001fc: 000 001 002 003" in shown
        assert "XMEM row 1 starts at byte address 0x00000200" in shown
        assert "0x00000200: 004 005 006 007" in shown


class TestXmemAddressing:
    def test_a_row_address_from_cr_and_lr_in_a_narrow_format(self):
        state = IpuState()
        state.regfile.set_cr(4, 2)
        state.regfile.set_lr(0, 1)
        state.xmem.write_address(3 * 128, struct.pack("<2f", 1.5, -2.25))
        shown = show(state, "row cr4 lr0 2 f32")
        assert "XMEM row 3 starts at byte address 0x00000180" in shown
        assert f"{1.5:>15.9g} {-2.25:>15.9g}" in shown

    def test_a_row_is_512_bytes_in_fp32(self):
        state = new_state(fmt=ElementFormat.FP32)
        state.xmem.write_address(2 * 512, struct.pack("<2I", 1, 0xDEADBEEF))
        shown = show(state, "row 1 1 2 u32")
        assert "XMEM row 2 starts at byte address 0x00000400" in shown
        assert f"{1:010d} {0xDEADBEEF:010d}" in shown

    def test_a_byte_address_from_registers(self):
        state = IpuState()
        state.regfile.set_cr(3, 0x200)
        state.regfile.set_lr(2, 4)
        state.xmem.write_address(0x204, struct.pack("<I", 42))
        assert "  0x00000204: 0000000042" in show(state, "byte cr3 lr2 1 u32")

    def test_the_request_and_its_operands_are_case_insensitive(self):
        state = IpuState()
        state.xmem.write_address(0, struct.pack("<f", 3.5))
        assert "3.5" in show(state, "XMEM ROW CR0 LR0 1 F32")

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("byte 1 0 1 f32", "4-byte aligned"),
            ("byte 1 0 1 cell16", "2-byte aligned"),
            ("byte 0 0 0 hex", "positive integer"),
            ("byte 0 0 1 invalid", "format must"),
            ("row rx0 0 1 hex", "Invalid address term"),
            (f"byte {XMEM_SIZE_BYTES - 1} 0 2 hex", "out of bounds"),
            (f"row {XMEM_SIZE_BYTES // 512} 0 1 hex", "out of range"),
        ],
    )
    def test_an_invalid_request_is_a_debug_error(self, text, message):
        with pytest.raises(DebugError, match=message):
            show(IpuState(), text)


class TestCell16:
    def test_the_visible_alphabet_has_one_distinct_character_per_byte(self):
        assert len(VISIBLE_ALPHABET) == 256
        assert len(set(VISIBLE_ALPHABET)) == 256

    def test_a_cell_decodes_to_its_character_and_colours(self):
        assert decode_16bit_cell(200, 14, 4) == (VISIBLE_ALPHABET[200], 14, 4)

    def test_inputs_are_normalised_to_their_encoded_bit_widths(self):
        assert decode_16bit_cell(0x100, 0x11, 0x22) == (VISIBLE_ALPHABET[0], 1, 2)
