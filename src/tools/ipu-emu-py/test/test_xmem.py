"""Tests for ipu_emu.xmem — mirrors test/test_xmem.cpp."""

import pytest

from ipu_emu.xmem import XMem, align_addr, words_needed_for_bytes, XMEM_SIZE_BYTES


# ---------------------------------------------------------------------------
# align_addr / words_needed_for_bytes helpers
# ---------------------------------------------------------------------------

class TestAlignAddr:
    def test_already_aligned(self):
        assert align_addr(0) == 0
        assert align_addr(128) == 128
        assert align_addr(256) == 256

    def test_rounds_up(self):
        assert align_addr(1) == 128
        assert align_addr(127) == 128
        assert align_addr(129) == 256

    def test_words_needed(self):
        assert words_needed_for_bytes(0) == 0
        assert words_needed_for_bytes(1) == 1
        assert words_needed_for_bytes(128) == 1
        assert words_needed_for_bytes(129) == 2


# ---------------------------------------------------------------------------
# XMem — port of test_xmem.cpp tests
# ---------------------------------------------------------------------------

class TestXMem:
    def test_initialize_zeroed(self):
        """XmemInitialize.ZeroedMemory"""
        xmem = XMem()
        buf = xmem.read_address(0, 16)
        assert buf == bytearray(16)

    def test_basic_round_trip(self):
        """XmemWriteRead.BasicRoundTrip"""
        xmem = XMem()
        data = bytearray([1, 2, 3, 4])
        xmem.write_address(10, data)
        out = xmem.read_address(10, 4)
        assert out == data

    def test_load_array(self):
        """XmemLoadArray.LoadsCorrectly"""
        xmem = XMem()
        arr = bytearray([10, 11, 12, 13, 14])
        start = 200
        xmem.load_array_to(arr, start)
        out = xmem.read_address(start, 5)
        assert out == arr

    def test_load_matrix_row_alignment(self):
        """XmemLoadMatrix.RowAlignmentBehavior"""
        xmem = XMem()
        rows, cols = 2, 3
        matrix = bytearray(range(1, rows * cols + 1))
        start = 0
        xmem.load_matrix_to(matrix, rows, cols, start)

        # First row at start + align(0) = 0
        out0 = xmem.read_address(start + align_addr(0), cols)
        assert out0 == matrix[:cols]

        # Second row at start + align(3) = 128
        offset1 = align_addr(3)
        out1 = xmem.read_address(start + offset1, cols)
        assert out1 == matrix[cols:]

    def test_last_byte_access(self):
        """XmemBounds.LastByteAccess"""
        xmem = XMem()
        last = XMEM_SIZE_BYTES - 1
        xmem.write_address(last, bytes([0xAA]))
        out = xmem.read_address(last, 1)
        assert out[0] == 0xAA

    def test_read_out_of_bounds_raises(self):
        xmem = XMem()
        with pytest.raises(ValueError):
            xmem.read_address(XMEM_SIZE_BYTES, 1)

    def test_write_out_of_bounds_raises(self):
        xmem = XMem()
        with pytest.raises(ValueError):
            xmem.write_address(XMEM_SIZE_BYTES, b"\x00")

    def test_clear(self):
        xmem = XMem()
        xmem.write_address(0, bytes([0xFF] * 16))
        xmem.clear()
        assert xmem.read_address(0, 16) == bytearray(16)

    def test_len_and_size(self):
        xmem = XMem()
        assert len(xmem) == XMEM_SIZE_BYTES
        assert xmem.size == XMEM_SIZE_BYTES

    def test_getitem_setitem(self):
        xmem = XMem()
        xmem[42] = 0xBB
        assert xmem[42] == 0xBB

    def test_write_read_returns_copy(self):
        """Ensure read_address returns a copy, not a view."""
        xmem = XMem()
        xmem.write_address(0, bytes([1, 2, 3]))
        out = xmem.read_address(0, 3)
        out[0] = 99  # mutate the copy
        assert xmem.read_address(0, 1)[0] == 1  # original unchanged
