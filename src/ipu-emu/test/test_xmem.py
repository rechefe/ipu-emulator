"""Tests for ipu_emu.xmem."""

import pytest

from ipu_emu.xmem import XMEM_SIZE_BYTES, XMem

# ---------------------------------------------------------------------------
# XMem
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


# ---------------------------------------------------------------------------
# Metadata side store (ADR 0007): one 16-bit word per 128-byte unit
# ---------------------------------------------------------------------------


class TestXMemMetadata:
    def test_unwritten_metadata_reads_zero(self):
        xmem = XMem()
        assert xmem.read_metadata(0) == 0
        assert xmem.read_metadata(XMEM_SIZE_BYTES - 128) == 0
        assert list(xmem.touched_metadata()) == []

    def test_write_then_read_per_unit(self):
        xmem = XMem()
        xmem.write_metadata(256, 0x1FF)
        xmem.write_metadata(XMEM_SIZE_BYTES - 128, 0xFFFF)
        assert xmem.read_metadata(256) == 0x1FF
        assert xmem.read_metadata(128) == 0 and xmem.read_metadata(384) == 0
        assert xmem.snapshot_metadata(XMEM_SIZE_BYTES - 128) == 0xFFFF
        assert list(xmem.touched_metadata()) == [(256, 0x1FF), (XMEM_SIZE_BYTES - 128, 0xFFFF)]

    def test_metadata_and_bytes_are_independent(self):
        xmem = XMem()
        xmem.write_address(128, b"\x07" * 128)
        assert xmem.read_metadata(128) == 0
        xmem.write_metadata(128, 3)
        assert xmem.read_address(128, 128) == b"\x07" * 128
        assert list(xmem.touched_pages())  # bytes stored; the metadata write added none

    def test_writing_zero_forgets_the_word(self):
        xmem = XMem()
        xmem.write_metadata(0, 5)
        xmem.write_metadata(0, 0)
        assert list(xmem.touched_metadata()) == []

    @pytest.mark.parametrize("address", [1, 64, -128, XMEM_SIZE_BYTES])
    def test_an_unaligned_or_out_of_range_unit_raises(self, address):
        xmem = XMem()
        with pytest.raises(ValueError):
            xmem.write_metadata(address, 1)
        with pytest.raises(ValueError):
            xmem.read_metadata(address)

    @pytest.mark.parametrize("word", [-1, 0x10000, True, 1.0])
    def test_a_word_that_is_not_16_bit_unsigned_raises(self, word):
        with pytest.raises(ValueError):
            XMem().write_metadata(0, word)

    def test_clear_drops_metadata(self):
        xmem = XMem()
        xmem.write_metadata(128, 9)
        xmem.clear()
        assert xmem.read_metadata(128) == 0
        assert list(xmem.touched_metadata()) == []

    def test_fill_leaves_metadata(self):
        xmem = XMem()
        xmem.write_metadata(0, 9)
        xmem.fill(0, PAGE_BYTES)
        assert xmem.read_metadata(0) == 9

    def test_eq_compares_metadata(self):
        a, b = XMem(), XMem()
        a.write_metadata(128, 4)
        assert a != b
        b.write_metadata(128, 4)
        assert a == b
        a.write_metadata(128, 0)  # zero equals never written
        b.write_metadata(128, 0)
        a.write_metadata(256, 0)
        assert a == b

    def test_observers_see_distinct_metadata_events(self):
        xmem = XMem()
        events = []
        xmem.add_observer(lambda *event: events.append(event))
        xmem.write_metadata(384, 0x0102)
        assert xmem.read_metadata(384) == 0x0102
        assert xmem.snapshot_metadata(384) == 0x0102  # no event
        assert events == [
            ("meta_write", 384, 2, b"\x02\x01"),
            ("meta_read", 384, 2, b"\x02\x01"),
        ]
