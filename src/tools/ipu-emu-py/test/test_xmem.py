"""Tests for ipu_emu.xmem — mirrors test/test_xmem.cpp."""

import pytest

from ipu_emu.xmem import XMEM_SIZE_BYTES, XMem, align_addr, words_needed_for_bytes

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

    def test_dense_descriptor_capacity(self):
        """The 300 MiB descriptor output must fit, far beyond the old 8 MB allocation."""
        assert XMEM_SIZE_BYTES == 512 * 1024**2
        xmem = XMem()
        address = 326 * 1024**2
        xmem.write_address(address, b"descriptor")
        assert xmem.read_address(address, 10) == b"descriptor"

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


# ---------------------------------------------------------------------------
# Ranged writes through indexing
# ---------------------------------------------------------------------------


class TestXMemIndexing:
    def test_contiguous_slice_write(self):
        xmem = XMem()
        xmem[256:260] = b"abcd"
        assert xmem.read_address(256, 4) == b"abcd"

    def test_strided_setitem_raises(self):
        xmem = XMem()
        with pytest.raises(TypeError):
            xmem[0:8:2] = b"abcd"

    def test_slice_write_cannot_resize(self):
        xmem = XMem()
        with pytest.raises(ValueError):
            xmem[0:4] = b"ab"


# ---------------------------------------------------------------------------
# Registered observers and ranged events
# ---------------------------------------------------------------------------


class TestXMemObservers:
    def test_two_observers_both_receive_events(self):
        xmem = XMem()
        first, second = [], []
        xmem.add_observer(lambda *event: first.append(event))
        xmem.add_observer(lambda *event: second.append(event))
        xmem.write_address(64, b"ab")
        assert first == second == [("write", 64, 2, b"ab")]

    def test_remover_detaches(self):
        xmem = XMem()
        events = []
        remove = xmem.add_observer(lambda *event: events.append(event))
        xmem.read_address(0, 4)
        remove()
        remove()  # idempotent
        xmem.read_address(0, 4)
        assert events == [("read", 0, 4, bytearray(4))]

    def test_one_byte_write_is_one_event_of_size_one(self):
        xmem = XMem()
        events = []
        xmem.add_observer(lambda *event: events.append(event))
        xmem[1000] = 7
        assert [(kind, address, size) for kind, address, size, _ in events] == [("write", 1000, 1)]

    def test_slice_write_reports_its_range(self):
        xmem = XMem()
        events = []
        xmem.add_observer(lambda *event: events.append(event))
        xmem[300:304] = b"wxyz"
        assert events == [("write", 300, 4, b"wxyz")]

    def test_clear_is_one_ranged_event(self):
        xmem = XMem()
        events = []
        xmem.add_observer(lambda *event: events.append(event))
        xmem.clear()
        assert events == [("clear", 0, XMEM_SIZE_BYTES, None)]


# ---------------------------------------------------------------------------
# Paged sparse storage
# ---------------------------------------------------------------------------

from ipu_emu.xmem import PAGE_BYTES  # noqa: E402


class TestXMemPaged:
    def test_page_holds_whole_rows(self):
        assert PAGE_BYTES % 512 == 0
        assert XMEM_SIZE_BYTES % PAGE_BYTES == 0

    def test_untouched_address_reads_zero_without_allocating(self):
        xmem = XMem()
        assert xmem.read_address(300 * 1024**2, 512) == bytearray(512)
        assert xmem[XMEM_SIZE_BYTES - 1] == 0
        assert list(xmem.touched_pages()) == []

    def test_write_across_page_boundary_reads_back(self):
        xmem = XMem()
        data = bytes(range(1, 201))
        address = 3 * PAGE_BYTES - 100
        xmem.write_address(address, data)
        assert xmem.read_address(address, len(data)) == data
        assert xmem[address : address + len(data)] == data
        assert [base for base, _ in xmem.touched_pages()] == [2 * PAGE_BYTES, 3 * PAGE_BYTES]

    def test_zero_write_to_absent_page_allocates_nothing(self):
        xmem = XMem()
        xmem.write_address(PAGE_BYTES, bytes(1024))
        assert list(xmem.touched_pages()) == []

    def test_clear_drops_every_page(self):
        xmem = XMem()
        for page in range(4):
            xmem.write_address(page * PAGE_BYTES + 7, b"\x01")
        xmem.clear()
        assert list(xmem.touched_pages()) == []
        assert xmem.read_address(7, 1) == b"\x00"

    def test_eq_treats_absent_and_zero_pages_as_equal(self):
        a, b = XMem(), XMem()
        a.write_address(5 * PAGE_BYTES, b"\x01")
        a.write_address(5 * PAGE_BYTES, b"\x00")  # page allocated, all zero
        assert len(list(a.touched_pages())) == 1
        assert a == b and b == a
        b.write_address(9, b"\x02")
        assert a != b
        a.write_address(9, b"\x02")
        assert a == b

    def test_fill(self):
        xmem = XMem()
        xmem.fill(PAGE_BYTES - 2, 4, 0xAB)
        assert xmem.read_address(PAGE_BYTES - 3, 6) == b"\x00" + b"\xab" * 4 + b"\x00"
        xmem.fill(0, 2 * PAGE_BYTES)
        assert list(xmem.touched_pages()) == []

    def test_snapshot_bytes_has_no_observer_event(self):
        xmem = XMem()
        xmem.write_address(10, b"xyz")
        events = []
        xmem.add_observer(lambda *event: events.append(event))
        assert xmem.snapshot_bytes(10, 3) == b"xyz"
        assert events == []

    def test_touched_pages_are_read_only(self):
        xmem = XMem()
        xmem.write_address(0, b"\x01")
        ((_, view),) = xmem.touched_pages()
        with pytest.raises(TypeError):
            view[0] = 2

    def test_full_range_slice_raises(self):
        xmem = XMem()
        with pytest.raises(ValueError, match="touched_pages"):
            xmem[:]
