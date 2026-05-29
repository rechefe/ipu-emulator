"""Profiling wrapper for XMEM that tracks peak memory lookahead per CR region.

For each data region (identified by CR base register index), computes:

    peak_lookahead_rows = max over all accesses of:
        (max_row_seen_so_far_in_region - current_access_row)

This answers: "how many rows back must we keep buffered at peak?"
"""

from __future__ import annotations

from ipu_emu.xmem import XMem

ROW_BYTES = 128  # one XMEM row = 128 bytes


class ProfilingXMem(XMem):
    """XMem subclass that records peak lookahead distance per CR region.

    Usage::

        profiling_xmem = ProfilingXMem()
        state.xmem = profiling_xmem   # swap into IpuState before running

        # After run:
        results = profiling_xmem.results()  # {cr_idx: peak_lookahead_rows}
    """

    def __init__(self) -> None:
        super().__init__()
        self._max_row: dict[int, int] = {}        # cr_idx → highest row seen so far
        self._peak_lookahead: dict[int, int] = {} # cr_idx → answer

    def record_access(self, cr_idx: int, addr: int) -> None:
        """Record an XMEM access at *addr* belonging to region *cr_idx*.

        Updates peak lookahead:
            lookahead = max_row_so_far - current_row
        """
        row = addr // ROW_BYTES
        prev_max = self._max_row.get(cr_idx, row)
        new_max = max(prev_max, row)
        self._max_row[cr_idx] = new_max
        lookahead = new_max - row
        if lookahead > self._peak_lookahead.get(cr_idx, 0):
            self._peak_lookahead[cr_idx] = lookahead

    def results(self) -> dict[int, int]:
        """Return {cr_idx: peak_lookahead_rows} for all regions accessed.

        Regions accessed only sequentially (no lookahead) appear with value 0.
        """
        return {cr_idx: self._peak_lookahead.get(cr_idx, 0) for cr_idx in self._max_row}
