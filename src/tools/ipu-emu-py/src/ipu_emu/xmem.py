"""External memory (XMEM) model.

A flat 512 MiB byte-addressable memory with the C ``xmem__obj_t`` interface.
Supports 128-byte word alignment helpers and bulk load utilities.

Allocation is mode-independent: XMEM is always 512 MiB, in both narrow and
wide-vector debug mode. Which of it is *reachable* depends on the active
mode's row size and is enforced by the caller (``Ipu._xmem_row_addr``),
not here — this class has no notion of "mode".
"""

from __future__ import annotations

from typing import Callable

#: ``fn(kind, address, size, data)`` with *kind* one of ``"read"``,
#: ``"write"`` or ``"clear"``. The event covers exactly the bytes
#: ``[address, address + size)`` the operation touched. *data* holds those
#: bytes for reads and writes, and is ``None`` for ``"clear"`` and ``fill``.
MemoryObserver = Callable[[str, int, int, "bytes | bytearray | None"], None]

XMEM_SIZE_BYTES = 1 << 29        # 512 MiB; full-resolution FP32 descriptors fit
XMEM_WIDTH_BYTES = 128           # one "word" = 128 bytes
XMEM_DEPTH_WORDS = XMEM_SIZE_BYTES // XMEM_WIDTH_BYTES


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


def _without(observers: tuple, fn: Callable) -> tuple:
    """*observers* with the first occurrence of *fn* removed (if any)."""
    for i, observer in enumerate(observers):
        if observer is fn:
            return observers[:i] + observers[i + 1:]
    return observers


class XMem:
    """512 MiB flat byte-addressable external memory.

    Internally stored as a ``bytearray`` for efficient byte-level access.
    """

    def __init__(self) -> None:
        self._data = bytearray(XMEM_SIZE_BYTES)
        self._constant_ones: dict[int, bytes] = {}
        self._observers: tuple[MemoryObserver, ...] = ()

    def add_observer(self, fn: MemoryObserver) -> Callable[[], None]:
        """Report every XMEM access to *fn*; returns a function that detaches it."""
        self._observers = self._observers + (fn,)
        return lambda: setattr(self, "_observers", _without(self._observers, fn))

    def _notify(self, kind: str, address: int, size: int, data) -> None:
        for observer in self._observers:
            observer(kind, address, size, data)

    def mark_constant_ones(self, address: int, data: bytes) -> None:
        """Record host-declared ONES provenance, after a validated write."""
        if self.read_address(address, len(data)) != data:
            raise ValueError("constant ONES declaration does not match memory")
        self._constant_ones[address] = data

    def is_constant_ones(self, address: int, data: bytes | bytearray) -> bool:
        return self._constant_ones.get(address) == data

    def _invalidate_constants(self, start: int, end: int) -> None:
        if self._constant_ones:
            self._constant_ones = {a: d for a, d in self._constant_ones.items()
                                   if a + len(d) <= start or a >= end}

    # -- low-level access ---------------------------------------------------

    def read_address(self, address: int, size: int) -> bytearray:
        """Read *size* bytes starting at *address*.  Returns a *copy*."""
        if address < 0 or address + size > XMEM_SIZE_BYTES:
            raise ValueError(
                f"XMEM read out of bounds: address={address}, size={size}, "
                f"end={address + size}, max={XMEM_SIZE_BYTES}"
            )
        result = bytearray(self._data[address : address + size])
        self._notify("read", address, size, result)
        return result

    def write_address(self, address: int, data: bytes | bytearray) -> None:
        """Write *data* starting at *address*."""
        size = len(data)
        if address < 0 or address + size > XMEM_SIZE_BYTES:
            raise ValueError(
                f"XMEM write out of bounds: address={address}, size={size}, "
                f"end={address + size}, max={XMEM_SIZE_BYTES}"
            )
        self._invalidate_constants(address, address + size)
        self._data[address : address + size] = data
        self._notify("write", address, size, data)

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
        """Zero the entire memory."""
        self._constant_ones.clear()
        self._data[:] = bytes(XMEM_SIZE_BYTES)
        self._notify("clear", 0, XMEM_SIZE_BYTES, None)

    def __getitem__(self, idx: int | slice) -> int | bytearray:
        return self._data[idx]

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
