"""External memory (XMEM) model.

A flat 2 MB byte-addressable memory that mirrors the C ``xmem__obj_t``.
Supports 128-byte word alignment helpers and bulk load utilities.
"""

from __future__ import annotations

XMEM_SIZE_BYTES = 1 << 21        # 2 MB
XMEM_WIDTH_BYTES = 128           # one "word" = 128 bytes
XMEM_DEPTH_WORDS = XMEM_SIZE_BYTES // XMEM_WIDTH_BYTES


def align_addr(addr: int) -> int:
    """Round *addr* up to the next 128-byte boundary (no-op if already aligned)."""
    rem = addr % XMEM_WIDTH_BYTES
    return addr if rem == 0 else addr + XMEM_WIDTH_BYTES - rem


def words_needed_for_bytes(n: int) -> int:
    """Number of 128-byte words required to hold *n* bytes."""
    return align_addr(n) // XMEM_WIDTH_BYTES


class XMem:
    """2 MB flat byte-addressable external memory.

    Internally stored as a ``bytearray`` for efficient byte-level access.
    """

    def __init__(self) -> None:
        self._data = bytearray(XMEM_SIZE_BYTES)

    # -- low-level access ---------------------------------------------------

    def read_address(self, address: int, size: int) -> bytearray:
        """Read *size* bytes starting at *address*.  Returns a *copy*."""
        if address < 0 or address + size > XMEM_SIZE_BYTES:
            raise ValueError(
                f"XMEM read out of bounds: address={address}, size={size}, "
                f"end={address + size}, max={XMEM_SIZE_BYTES}"
            )
        return bytearray(self._data[address : address + size])

    def write_address(self, address: int, data: bytes | bytearray) -> None:
        """Write *data* starting at *address*."""
        size = len(data)
        if address < 0 or address + size > XMEM_SIZE_BYTES:
            raise ValueError(
                f"XMEM write out of bounds: address={address}, size={size}, "
                f"end={address + size}, max={XMEM_SIZE_BYTES}"
            )
        self._data[address : address + size] = data

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
        self._data[:] = b"\x00" * XMEM_SIZE_BYTES

    def __getitem__(self, idx: int | slice) -> int | bytearray:
        return self._data[idx]

    def __setitem__(self, idx: int | slice, val: int | bytes | bytearray) -> None:
        self._data[idx] = val

    def __len__(self) -> int:
        return XMEM_SIZE_BYTES

    def __repr__(self) -> str:
        return f"XMem(size={XMEM_SIZE_BYTES})"
