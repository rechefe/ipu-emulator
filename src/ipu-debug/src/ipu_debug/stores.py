"""The row the last AaQ store wrote, and XMEM row metadata as the debugger shows it.

Since ADR 0007 the AaQ stage stores its own result: ``ACTIVATE.QUANTIZE``
writes one row (128 NX8 bytes and the row's metadata word, or 512 FP32 bytes)
and ``ACTIVATE.EXACT`` one FP32 row. No register holds that row any more (the
old ``POST_AAQ_REG``), so the debugger remembers which row was written last:
an XMEM observer, one per memory, records every whole-row write. Only the
AaQ slot writes whole rows while a program runs; the debugger's own edits
write single values.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

from ipu_emu.ipu import xmem_row_size_bytes
from ipu_emu.ipu_state import IpuState
from ipu_emu.xmem import METADATA_UNIT_BYTES, XMem
from ipu_isa.nx8 import SCALE_BIAS, split_metadata

__all__ = ["AaqRow", "last_aaq_row", "metadata_text", "track_aaq_stores"]


@dataclass(frozen=True)
class AaqRow:
    """The row the last AaQ store wrote, as XMEM holds it now.

    Attributes:
        address: Its first byte, or ``None`` before the first store.
        data: Its bytes (zeros before the first store).
        metadata: Its metadata word in NX8 mode; ``None`` in FP32 mode.
    """

    address: int | None
    data: bytes
    metadata: int | None

    @property
    def row(self) -> int | None:
        """Its row number in the mode's row size, or ``None``."""
        return None if self.address is None else self.address // len(self.data)


class _StoreTracker:
    """Remembers the address of the last whole-row XMEM write."""

    def __init__(self) -> None:
        self.last: int | None = None

    def observe(self, kind: str, address: int, size: int, data: object) -> None:
        del data
        if kind == "write" and size in (METADATA_UNIT_BYTES, 4 * METADATA_UNIT_BYTES):
            if address % size == 0:
                self.last = address


#: One tracker per live memory, by identity (``XMem`` compares by value, so it
#: is no dictionary key); an entry goes when its memory is collected.
_TRACKERS: dict[int, _StoreTracker] = {}


def _tracker(xmem: XMem) -> _StoreTracker | None:
    return _TRACKERS.get(id(xmem))


def track_aaq_stores(state: IpuState) -> None:
    """Start remembering *state*'s AaQ stores (once per memory; later calls do nothing)."""
    if _tracker(state.xmem) is None:
        tracker = _StoreTracker()
        state.xmem.add_observer(tracker.observe)
        _TRACKERS[id(state.xmem)] = tracker
        weakref.finalize(state.xmem, _TRACKERS.pop, id(state.xmem), None)


def last_aaq_row(state: IpuState) -> AaqRow:
    """The row the last AaQ store wrote since :func:`track_aaq_stores`."""
    row_bytes = xmem_row_size_bytes(state)
    tracker = _tracker(state.xmem)
    address = None if tracker is None else tracker.last
    if address is None or address % row_bytes:
        return AaqRow(None, bytes(row_bytes), None)
    data = state.xmem.snapshot_bytes(address, row_bytes)
    nx8 = row_bytes == METADATA_UNIT_BYTES
    return AaqRow(address, data, state.xmem.snapshot_metadata(address) if nx8 else None)


def metadata_text(word: int) -> str:
    """A metadata word as the panes show it: ``scale 2^5``, plus
    ``dynamic exponent 1`` when that bit is set (ADR 0007)."""
    scale, dynamic_exponent = split_metadata(word)
    extra = ", dynamic exponent 1" if dynamic_exponent else ""
    return f"scale 2^{scale - SCALE_BIAS}{extra}"
