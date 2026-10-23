"""External memory (XMEM) model.

A flat 512 MiB byte-addressable memory: address reads and writes, observer
events, and constant-ONES provenance. Storage is sparse: only pages that have
been written are allocated.

Beside the bytes, XMEM keeps one 16-bit metadata word per 128-byte unit (an
NX8 row's ``scale`` and ``dynamic_exponent``, ADR 0007). The side store is
sparse too: a unit whose metadata was never written reads 0.

The address space is the same in every element format: XMEM is always 512 MiB.
Which of it is *reachable* depends on the active format's row size and is
enforced by the caller (``ipu_emu.geometry.xmem_row_addr``), not here — this
class has no notion of element format.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator

from ipu_emu.errors import ConfigurationError, XMemBoundsError
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import XMEM_BYTES

#: ``fn(kind, address, size, data)`` with *kind* one of ``"read"``,
#: ``"write"`` or ``"clear"``. The event covers exactly the bytes
#: ``[address, address + size)`` the operation touched. *data* holds those
#: bytes for reads and writes, and is ``None`` for ``"clear"`` and ``fill``.
#: The metadata side store reports ``"meta_read"`` and ``"meta_write"``:
#: *address* is the unit's first byte, *size* is ``METADATA_WORD_BYTES`` and
#: *data* the word, little-endian. An observer ignores kinds it does not know.
MemoryObserver = Callable[[str, int, int, "bytes | bytearray | None"], None]

XMEM_SIZE_BYTES = XMEM_BYTES  # 512 MiB; full-resolution FP32 descriptors fit
XMEM_WIDTH_BYTES = FORMATS[ElementFormat.INT8].row_bytes  # one "word": the narrow row

#: The XMEM bytes one metadata word describes: one NX8 row (ADR 0007).
METADATA_UNIT_BYTES = 128
#: The size of one metadata word.
METADATA_WORD_BYTES = 2
_METADATA_WORD_MAX = (1 << (8 * METADATA_WORD_BYTES)) - 1

# Storage granule. A multiple of the widest row (512 bytes in the FP32/INT32
# formats), so an aligned row access never crosses a page.
PAGE_BYTES = 1 << 16
_ZERO_PAGE = bytes(PAGE_BYTES)
_ZERO_PAGE_VIEW = memoryview(_ZERO_PAGE)


def _contiguous_range(idx: slice) -> tuple[int, int]:
    """Resolve a step-1 slice to ``(start, stop)``; strided slices raise."""
    if idx.step not in (None, 1):
        raise TypeError(f"XMEM slices must be contiguous (step 1), got step {idx.step}")
    start, stop, _ = idx.indices(XMEM_SIZE_BYTES)
    return start, max(start, stop)


def _word_bytes(word: int) -> bytes:
    """A metadata word as an observer event's data: little-endian bytes."""
    return word.to_bytes(METADATA_WORD_BYTES, "little")


def _remover(owner, fn: Callable) -> Callable[[], None]:
    """A function that detaches *fn* from *owner*. It holds *owner* weakly, so
    an observer that keeps its remover does not form a reference cycle."""
    ref = weakref.ref(owner)

    def remove() -> None:
        target = ref()
        if target is not None:
            target._observers = _without(target._observers, fn)

    return remove


def _without(observers: tuple, fn: Callable) -> tuple:
    """*observers* with the first occurrence of *fn* removed (if any)."""
    for i, observer in enumerate(observers):
        if observer is fn:
            return observers[:i] + observers[i + 1 :]
    return observers


class XMem:
    """512 MiB byte-addressable external memory, allocated a page at a time.

    The whole address space is addressable, but only pages that have been
    written to hold storage (``PAGE_BYTES`` each); a missing page reads as
    zeros. Accesses that cross a page boundary are split internally.

    Observers registered with :meth:`add_observer` see one ranged event per
    operation, covering exactly the bytes it touched.
    """

    def __init__(self) -> None:
        self._pages: dict[int, bytearray] = {}
        self._constant_ones: dict[int, bytes] = {}
        # Metadata word by unit index (address // METADATA_UNIT_BYTES); a
        # zero word is never stored, so equal memories hold equal dicts.
        self._metadata: dict[int, int] = {}
        self._observers: tuple[MemoryObserver, ...] = ()

    def add_observer(self, fn: MemoryObserver) -> Callable[[], None]:
        """Report every XMEM access to *fn*; returns a function that detaches it."""
        self._observers = self._observers + (fn,)
        return _remover(self, fn)

    def _notify(self, kind: str, address: int, size: int, data) -> None:
        for observer in self._observers:
            observer(kind, address, size, data)

    def mark_constant_ones(self, address: int, data: bytes) -> None:
        """Record host-declared ONES provenance, after a validated write."""
        if self.snapshot_bytes(address, len(data)) != data:
            raise ConfigurationError("constant ONES declaration does not match memory")
        self._constant_ones[address] = data

    def is_constant_ones(self, address: int, data: bytes | bytearray) -> bool:
        return self._constant_ones.get(address) == data

    def _invalidate_constants(self, start: int, end: int) -> None:
        if self._constant_ones:
            self._constant_ones = {
                a: d for a, d in self._constant_ones.items() if a + len(d) <= start or a >= end
            }

    # -- paged storage ------------------------------------------------------

    @staticmethod
    def _check_bounds(kind: str, address: int, size: int) -> None:
        if address < 0 or size < 0 or address + size > XMEM_SIZE_BYTES:
            raise XMemBoundsError(
                f"XMEM {kind} out of bounds: address={address}, size={size}, "
                f"end={address + size}, max={XMEM_SIZE_BYTES}"
            )

    def _read(self, address: int, size: int) -> bytearray:
        page, offset = divmod(address, PAGE_BYTES)
        if offset + size <= PAGE_BYTES:  # the common case: one row, one page
            buf = self._pages.get(page)
            return bytearray(size) if buf is None else buf[offset : offset + size]
        out = bytearray(size)
        pos = 0
        while pos < size:
            page, offset = divmod(address + pos, PAGE_BYTES)
            n = min(PAGE_BYTES - offset, size - pos)
            buf = self._pages.get(page)
            if buf is not None:
                out[pos : pos + n] = memoryview(buf)[offset : offset + n]
            pos += n
        return out

    def _write(self, address: int, data: bytes | bytearray | memoryview) -> None:
        try:
            view = memoryview(data)
        except TypeError:  # an iterable of ints, as bytearray slicing accepts
            view = memoryview(bytes(data))
        if view.format != "B" or view.ndim != 1:
            view = view.cast("B")
        size = len(view)
        pos = 0
        while pos < size:
            page, offset = divmod(address + pos, PAGE_BYTES)
            n = min(PAGE_BYTES - offset, size - pos)
            chunk = view[pos : pos + n]
            buf = self._pages.get(page)
            if buf is None:
                if _ZERO_PAGE.startswith(chunk):
                    pos += n
                    continue  # zeros into an absent page: nothing to store
                buf = self._pages[page] = bytearray(PAGE_BYTES)
            buf[offset : offset + n] = chunk
            pos += n

    # -- low-level access ---------------------------------------------------

    def read_address(self, address: int, size: int) -> bytearray:
        """Read *size* bytes starting at *address*.  Returns a *copy*."""
        self._check_bounds("read", address, size)
        result = self._read(address, size)
        self._notify("read", address, size, result)
        return result

    def write_address(self, address: int, data: bytes | bytearray) -> None:
        """Write *data* starting at *address*."""
        size = len(data)
        self._check_bounds("write", address, size)
        self._invalidate_constants(address, address + size)
        self._write(address, data)
        self._notify("write", address, size, data)

    def fill(self, address: int, size: int, value: int = 0) -> None:
        """Set *size* bytes from *address* to *value*.

        Filling with zero releases every page the range fully covers. The
        observer event carries ``data=None``: the range is ``value`` repeated.
        """
        self._check_bounds("write", address, size)
        if not 0 <= value <= 0xFF:
            raise ConfigurationError(f"XMEM fill value must be a byte, got {value}")
        self._invalidate_constants(address, address + size)
        pos = 0
        while pos < size:
            page, offset = divmod(address + pos, PAGE_BYTES)
            n = min(PAGE_BYTES - offset, size - pos)
            if value == 0 and n == PAGE_BYTES:
                self._pages.pop(page, None)
            elif value == 0:
                buf = self._pages.get(page)
                if buf is not None:
                    buf[offset : offset + n] = _ZERO_PAGE_VIEW[:n]
            else:
                buf = self._pages.get(page)
                if buf is None:
                    buf = self._pages[page] = bytearray(PAGE_BYTES)
                buf[offset : offset + n] = bytes((value,)) * n
            pos += n
        self._notify("write", address, size, None)

    # -- metadata side store ------------------------------------------------

    @staticmethod
    def _metadata_unit(address: int) -> int:
        XMem._check_bounds("metadata access", address, METADATA_UNIT_BYTES)
        if address % METADATA_UNIT_BYTES:
            raise ConfigurationError(
                f"XMEM metadata address {address} is not aligned to a "
                f"{METADATA_UNIT_BYTES}-byte unit"
            )
        return address // METADATA_UNIT_BYTES

    def read_metadata(self, address: int) -> int:
        """The metadata word of the 128-byte unit at *address* (aligned); 0
        when it was never written. Observers see a ``"meta_read"`` event."""
        word = self._metadata.get(self._metadata_unit(address), 0)
        self._notify("meta_read", address, METADATA_WORD_BYTES, _word_bytes(word))
        return word

    def write_metadata(self, address: int, word: int) -> None:
        """Set the metadata word of the 128-byte unit at *address* (aligned).
        Observers see a ``"meta_write"`` event; the unit's bytes are untouched.

        Raises:
            ConfigurationError: *address* is out of bounds or unaligned, or
                *word* is not an unsigned 16-bit int.
        """
        unit = self._metadata_unit(address)
        if isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= _METADATA_WORD_MAX:
            raise ConfigurationError(f"XMEM metadata takes an unsigned 16-bit int, got {word!r}")
        if word:
            self._metadata[unit] = word
        else:
            self._metadata.pop(unit, None)
        self._notify("meta_write", address, METADATA_WORD_BYTES, _word_bytes(word))

    def snapshot_metadata(self, address: int) -> int:
        """:meth:`read_metadata` without an observer event (for tools)."""
        return self._metadata.get(self._metadata_unit(address), 0)

    def touched_metadata(self) -> Iterator[tuple[int, int]]:
        """Yield ``(unit address, word)`` per nonzero metadata word, in address
        order. Every other unit's metadata is 0."""
        for unit in sorted(self._metadata):
            yield unit * METADATA_UNIT_BYTES, self._metadata[unit]

    # -- inspection (no observer events) ------------------------------------

    def snapshot_bytes(self, address: int, size: int) -> bytes:
        """Copy of *size* bytes at *address*, without an observer event.

        For debuggers and other tools that inspect memory without being part
        of the program's execution.
        """
        self._check_bounds("read", address, size)
        return bytes(self._read(address, size))

    def touched_pages(self) -> Iterator[tuple[int, memoryview]]:
        """Yield ``(base_address, read-only view)`` per allocated page, in
        address order. Every byte outside these pages is zero."""
        for page in sorted(self._pages):
            yield page * PAGE_BYTES, memoryview(self._pages[page]).toreadonly()

    def __eq__(self, other: object) -> bool:
        """Same bytes and metadata everywhere; an absent page equals a page of
        zeros, and unwritten metadata equals a zero word."""
        if not isinstance(other, XMem):
            return NotImplemented
        if self._metadata != other._metadata:
            return False
        mine, theirs = self._pages, other._pages
        for page in mine.keys() | theirs.keys():
            a = mine.get(page, _ZERO_PAGE)
            b = theirs.get(page, _ZERO_PAGE)
            if a != b:
                return False
        return True

    __hash__ = None  # mutable

    # -- convenience --------------------------------------------------------

    @property
    def size(self) -> int:
        return XMEM_SIZE_BYTES

    def clear(self) -> None:
        """Zero the entire memory, metadata included, by dropping every page."""
        self._constant_ones.clear()
        self._pages.clear()
        self._metadata.clear()
        self._notify("clear", 0, XMEM_SIZE_BYTES, None)

    def __getitem__(self, idx: int | slice) -> int | bytearray:
        """Read one byte or a contiguous range, without an observer event.

        Slicing the whole address space raises: it would materialise 512 MiB.
        Iterate :meth:`touched_pages` instead.
        """
        if isinstance(idx, slice):
            start, stop = _contiguous_range(idx)
            if stop - start == XMEM_SIZE_BYTES:
                raise ConfigurationError(
                    "slicing all of XMEM would allocate 512 MiB; iterate "
                    "XMem.touched_pages() (absent pages are zero) or use "
                    "XMem.__eq__ to compare memories"
                )
            return self._read(start, stop - start)
        if not isinstance(idx, int):
            raise TypeError(f"XMEM indices must be int or slice, not {type(idx).__name__}")
        if idx < 0:
            idx += XMEM_SIZE_BYTES
        if not 0 <= idx < XMEM_SIZE_BYTES:
            raise IndexError(f"XMEM index out of range: {idx}")
        return self._read(idx, 1)[0]

    def __setitem__(self, idx: int | slice, val: int | bytes | bytearray) -> None:
        """Write one byte (int index) or a contiguous range (slice, step 1).

        A slice must be replaced by exactly as many bytes as it covers: XMEM
        has a fixed size and never shifts. Strided slices raise ``TypeError``.
        """
        if isinstance(idx, slice):
            start, stop = _contiguous_range(idx)
            data = bytes(val)
            if len(data) != stop - start:
                raise ConfigurationError(
                    f"XMEM slice assignment must keep the size: slice covers "
                    f"{stop - start} bytes, value has {len(data)}"
                )
            self.write_address(start, data)
        elif isinstance(idx, int):
            if idx < 0:
                idx += XMEM_SIZE_BYTES
            self.write_address(idx, bytes((val,)))
        else:
            raise TypeError(f"XMEM indices must be int or slice, not {type(idx).__name__}")

    def __len__(self) -> int:
        return XMEM_SIZE_BYTES

    def __repr__(self) -> str:
        return f"XMem(size={XMEM_SIZE_BYTES})"
