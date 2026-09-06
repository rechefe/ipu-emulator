"""External memory (XMEM) model.

A flat 8 MB byte-addressable memory that mirrors the C ``xmem__obj_t``.
Supports 128-byte word alignment helpers and bulk load utilities.

Allocation is mode-independent: XMEM is always 8 MB, in both narrow and
wide-vector debug mode. Which of it is *reachable* depends on the active
mode's row size and is enforced by the caller (``Ipu._xmem_row_addr``),
not here — this class has no notion of "mode".
"""

from __future__ import annotations

XMEM_SIZE_BYTES = 1 << 23        # 8 MB
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
    """8 MB flat byte-addressable external memory.

    Internally stored as a ``bytearray`` for efficient byte-level access.
    """

    def __init__(self) -> None:
        self._data = bytearray(XMEM_SIZE_BYTES)
        self._constant_ones: dict[int, bytes] = {}
        self._profile_constants: dict[int, bytes] = {}
        self._profile_observer = None

    def mark_profile_constant(self, address: int, data: bytes) -> None:
        """Declare a constant/mask for profiling; subsequent writes revoke it."""
        data = bytes(data)
        if not data or self.read_address(address, len(data)) != data:
            raise ValueError("profile constant declaration does not match memory")
        self._profile_constants[address] = data

    def write_profile_constant(self, address: int, data: bytes) -> None:
        """Initialize a known constant/mask and declare its profiling provenance."""
        self.write_address(address, data)
        self.mark_profile_constant(address, data)

    def is_profile_constant(self, address: int, data: bytes) -> bool:
        return self._profile_constants.get(address) == data or self.is_constant_ones(address, data)

    def mark_constant_ones(self, address: int, data: bytes) -> None:
        """Record host-declared ONES provenance, after a validated write."""
        if self.read_address(address, len(data)) != data:
            raise ValueError("constant ONES declaration does not match memory")
        self._constant_ones[address] = data

    def is_constant_ones(self, address: int, data: bytes | bytearray) -> bool:
        return self._constant_ones.get(address) == data

    def _invalidate_constants(self, start: int, end: int) -> None:
        self._profile_constants = {a: d for a, d in self._profile_constants.items()
                                   if a + len(d) <= start or a >= end}
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
        if self._profile_observer is not None:
            self._profile_observer("read", address, result)
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
        if self._profile_observer is not None:
            self._profile_observer("write", address, data)

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
        self._profile_constants.clear()
        if self._profile_observer is not None:
            self._profile_observer("write", 0, self._data)
        self._data[:] = b"\x00" * XMEM_SIZE_BYTES

    def __getitem__(self, idx: int | slice) -> int | bytearray:
        return self._data[idx]

    def __setitem__(self, idx: int | slice, val: int | bytes | bytearray) -> None:
        # Indexing supports arbitrary strides; conservatively drop declarations.
        self._constant_ones.clear()
        self._profile_constants.clear()
        if self._profile_observer is not None:
            self._profile_observer("write", 0, self._data)
        self._data[idx] = val

    def __len__(self) -> int:
        return XMEM_SIZE_BYTES

    def __repr__(self) -> str:
        return f"XMem(size={XMEM_SIZE_BYTES})"
