"""External memory (XMEM) model.

A flat 512 MiB byte-addressable memory with the C ``xmem__obj_t`` interface.
Supports 128-byte word alignment helpers and bulk load utilities. Storage is
sparse: only pages that have been written are allocated.

The address space is mode-independent: XMEM is always 512 MiB, in both narrow and
wide-vector debug mode. Which of it is *reachable* depends on the active
mode's row size and is enforced by the caller (``Ipu._xmem_row_addr``),
not here — this class has no notion of "mode".
"""

from __future__ import annotations

import weakref

from typing import Callable, Iterator

#: ``fn(kind, address, size, data)`` with *kind* one of ``"read"``,
#: ``"write"`` or ``"clear"``. The event covers exactly the bytes
#: ``[address, address + size)`` the operation touched. *data* holds those
#: bytes for reads and writes, and is ``None`` for ``"clear"`` and ``fill``.
MemoryObserver = Callable[[str, int, int, "bytes | bytearray | None"], None]

XMEM_SIZE_BYTES = 1 << 29  # 512 MiB; full-resolution FP32 descriptors fit
XMEM_WIDTH_BYTES = 128  # one "word" = 128 bytes
XMEM_DEPTH_WORDS = XMEM_SIZE_BYTES // XMEM_WIDTH_BYTES

# Storage granule. A multiple of the widest row (512 bytes in wide-vector
# mode), so an aligned row access never crosses a page.
PAGE_BYTES = 1 << 16
_ZERO_PAGE = bytes(PAGE_BYTES)
_ZERO_PAGE_VIEW = memoryview(_ZERO_PAGE)


def align_addr(addr: int) -> int:
    """Round *addr* up to the next 128-byte boundary (no-op if already aligned)."""
    rem = addr % XMEM_WIDTH_BYTES
    return addr if rem == 0 else addr + XMEM_WIDTH_BYTES - rem


def words_needed_for_bytes(n: int) -> int:
    """Number of 128-byte words required to hold *n* bytes."""
    return align_addr(n) // XMEM_WIDTH_BYTES


def _contiguous_range(idx: slice) -> tuple[int, int]:
    """Resolve a step-1 slice to ``(start, stop)``; strided slices raise."""
    if idx.step not in (None, 1):
        raise TypeError(f"XMEM slices must be contiguous (step 1), got step {idx.step}")
    start, stop, _ = idx.indices(XMEM_SIZE_BYTES)
    return start, max(start, stop)


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
            raise ValueError("constant ONES declaration does not match memory")
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
            raise ValueError(
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
            raise ValueError(f"XMEM fill value must be a byte, got {value}")
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
        """Same bytes everywhere; an absent page equals a page of zeros."""
        if not isinstance(other, XMem):
            return NotImplemented
        mine, theirs = self._pages, other._pages
        for page in mine.keys() | theirs.keys():
            a = mine.get(page, _ZERO_PAGE)
            b = theirs.get(page, _ZERO_PAGE)
            if a != b:
                return False
        return True

    __hash__ = None  # mutable

    # -- bulk helpers (mirror C API) ----------------------------------------

    def load_array_to(self, array: bytes | bytearray, start_address: int) -> None:
        """Copy *array* into XMEM starting at *start_address*."""
        self.write_address(start_address, array)

    def load_matrix_to(
        self,
        matrix: bytes | bytearray,
        rows: int,
        cols: int,
        start_address: int,
    ) -> None:
        """Load a row-major matrix with each row aligned to 128-byte boundary.

        Mirrors ``xmem__load_matrix_to`` from the C implementation.
        """
        for i in range(rows):
            row_start = i * cols
            row_data = matrix[row_start : row_start + cols]
            addr_offset = align_addr(i * cols)
            self.load_array_to(row_data, start_address + addr_offset)

    # -- convenience --------------------------------------------------------

    @property
    def size(self) -> int:
        return XMEM_SIZE_BYTES

    def clear(self) -> None:
        """Zero the entire memory by dropping every page."""
        self._constant_ones.clear()
        self._pages.clear()
        self._notify("clear", 0, XMEM_SIZE_BYTES, None)

    def __getitem__(self, idx: int | slice) -> int | bytearray:
        """Read one byte or a contiguous range, without an observer event.

        Slicing the whole address space raises: it would materialise 512 MiB.
        Iterate :meth:`touched_pages` instead.
        """
        if isinstance(idx, slice):
            start, stop = _contiguous_range(idx)
            if stop - start == XMEM_SIZE_BYTES:
                raise ValueError(
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
                raise ValueError(
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
