"""The single buffer behind a :class:`~ipu_emu.regfile.RegFile`.

Every register of a register file lives in one ``bytearray``, in schema
order; the file maps each register name to its writable ``memoryview`` slice.
Copying a whole register file (the cycle's snapshot) is then one buffer copy.
"""

from __future__ import annotations

from ipu_emu.descriptors import RegDescriptor
from ipu_emu.errors import RegisterSizeError

_LAYOUTS: dict[int, tuple[list[RegDescriptor], dict[str, tuple[int, int]]]] = {}


def layout_of(schema: list[RegDescriptor]) -> dict[str, tuple[int, int]]:
    """Each register's (offset, size) in one buffer, in schema order; one
    shared dict per schema (so ``is`` tells two files of one layout apart)."""
    entry = _LAYOUTS.get(id(schema))
    if entry is None or entry[0] is not schema:
        layout: dict[str, tuple[int, int]] = {}
        offset = 0
        for desc in schema:
            size = desc.size_bytes * desc.count
            layout[desc.name] = (offset, size)
            offset += size
        entry = (schema, layout)  # holds the schema, so its id is not reused
        _LAYOUTS[id(schema)] = entry
    return entry[1]


def slices(buffer: bytearray, layout: dict[str, tuple[int, int]]) -> dict[str, memoryview]:
    """Each register's writable view of *buffer*."""
    whole = memoryview(buffer)
    return {name: whole[offset : offset + size] for name, (offset, size) in layout.items()}


def readonly_views(storage: dict[str, memoryview]) -> dict[str, memoryview]:
    """A read-only view of each register in *storage*: the ones
    ``RegFile.view`` hands out. A register's slice never changes identity
    (the buffer is never reallocated), so each stays valid for the file's life."""
    return {name: view.toreadonly() for name, view in storage.items()}


def write(view: memoryview, start: int, stop: int, data: object) -> None:
    """``view[start:stop] = data`` as a ``bytearray`` slice assignment does it.

    A register is a fixed-size view, not a ``bytearray``: data of the slice's
    size is written in place; anything else goes through a ``bytearray`` of
    the slice first, so it is converted (or refused, with the same error) as
    the ``bytearray`` storage did. Data that would have resized a register's
    storage raises ``RegisterSizeError``.
    """
    if (type(data) is bytes or type(data) is bytearray) and len(data) == stop - start:
        view[start:stop] = data
        return
    part = bytearray(view[start:stop])
    size = len(part)
    part[:] = data  # type: ignore[call-overload]
    if len(part) != size:
        raise RegisterSizeError(f"{size}-byte register range written with {len(part)} bytes")
    view[start:stop] = part
