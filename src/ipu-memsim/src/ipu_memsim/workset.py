"""Working-set numbers of a trace: how much memory a kernel needs, per region.

They come from the trace alone, so they hold for any memory model:

- **footprint**: distinct rows the kernel touches. Memory beyond the sum of
  the footprints cannot help: everything is already resident.
- **peak live rows**: a row is *live* from its first access to its last. The
  peak number of rows live at the same time is the least memory with which a
  perfect loader (one that brings each row in just before its first use and
  drops it right after its last) never has to fetch a row twice.
- **look-back**: the furthest the kernel ever goes behind the highest row it
  has reached so far. It is the smallest safe ``jump_back`` for a region that
  is walked front to back.

Example: reads of rows 0 1 2 1 2 3 2 3 4, one per access. Footprint 5. Row
1 is live over accesses 1 to 3, row 2 over 2 to 6, row 3 over 5 to 7; rows 0
and 4 are touched once. At most 2 rows are live at once (rows 1 and 2 at
access 2, rows 2 and 3 at access 5). Look-back is 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipu_memsim.trace import READ, Trace

__all__ = ["RegionWorkingSet", "working_set"]


@dataclass(frozen=True)
class RegionWorkingSet:
    """One region's numbers (see the module docstring); sizes in rows.

    Attributes:
        name, role: The region.
        rows: The rows the kernel declares for it.
        footprint: Distinct rows touched.
        peak_live: The peak number of rows live at once.
        look_back: The largest distance behind the furthest row reached.
        reads, writes: Accesses.
    """

    name: str
    role: str
    rows: int
    footprint: int
    peak_live: int
    look_back: int
    reads: int
    writes: int


def _peak_live(rows: np.ndarray) -> int:
    """The peak number of rows between their first and last access."""
    if len(rows) == 0:
        return 0
    positions = np.arange(len(rows))
    _, first = np.unique(rows, return_index=True)
    # The last access of each row: the first one in the reversed order.
    _, last_reversed = np.unique(rows[::-1], return_index=True)
    last = len(rows) - 1 - last_reversed
    # +1 where a row becomes live, -1 just after its last access.
    change = np.zeros(len(rows) + 1, dtype=np.int64)
    np.add.at(change, positions[first], 1)
    np.add.at(change, last + 1, -1)
    return int(np.cumsum(change).max())


def working_set(trace: Trace) -> list[RegionWorkingSet]:
    """The working-set numbers of every region the trace touches."""
    result = []
    for index, region in enumerate(trace.header["regions"]):
        mine = trace.region == index
        if not mine.any():
            continue
        rows = trace.row[mine]
        furthest = np.maximum.accumulate(rows)
        reads = int((trace.op[mine] == READ).sum())
        result.append(
            RegionWorkingSet(
                name=region["name"],
                role=region["role"],
                rows=region["rows"],
                footprint=len(np.unique(rows)),
                peak_live=_peak_live(rows),
                look_back=int((furthest - rows).max()),
                reads=reads,
                writes=int(mine.sum()) - reads,
            )
        )
    return result
