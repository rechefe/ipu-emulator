"""Memory models: each decides, per access, how many cycles the IPU stalls.

The interface is four methods (:class:`MemoryModel`):

- ``start(trace)`` once, before the first access (reset state, read regions);
- ``access(t, op, row, region) -> stall cycles`` for every access, in order;
- ``finish(t)`` once the kernel has halted, and ``counters()`` for the report.

``t`` is the time the access really happens: its recorded cycle plus every
stall before it (see ``ipu_memsim.replay``).

**Every model in this file is a TOY MODEL.** They are simple, well-understood
shapes (always miss, first touch, busy banks, a textbook cache) that bracket
the answer and test the replay engine. None of them follows the hardware
specification; the one that does is ``ipu_memsim.cache_unit.CacheUnit``.

Rows are flat XMEM rows as the emulator addresses them. A hardware row is
1024 bits = 128 bytes, so sizes in rows convert as ``rows * 128`` bytes
(2 MB = 16384 rows).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ipu_memsim.trace import WRITE, Trace

__all__ = ["AllMiss", "Banked", "Cache", "Ideal", "MemoryModel", "PageFirstTouch"]


class MemoryModel:
    """The interface every memory model implements."""

    #: The name reports show.
    name = "model"

    def start(self, trace: Trace) -> None:
        """Forget everything; get ready for ``trace``'s first access."""

    def access(self, t: int, op: int, row: int, region: int) -> int:
        """One access at time ``t``: return the cycles the IPU stalls for it.

        Args:
            t: The cycle the access happens in, earlier stalls included.
            op: ``READ`` or ``WRITE``.
            row: The flat XMEM row.
            region: The index of the region holding the row, or ``OTHER``.
        """
        raise NotImplementedError

    def finish(self, t: int) -> None:
        """The kernel halted at time ``t``. A model may account for work that
        goes on afterwards (draining outputs); it cannot add stall cycles."""

    def counters(self) -> dict[str, Any]:
        """The model's own numbers (hits, misses, ...), for the report."""
        return {}


class Ideal(MemoryModel):
    """Perfect memory: nothing ever stalls. Replays to the emulator's cycle count."""

    name = "ideal"

    def access(self, t: int, op: int, row: int, region: int) -> int:
        return 0


class AllMiss(MemoryModel):
    """Worst case: every access stalls ``penalty`` cycles.

    Example: 4410 accesses at ``penalty = 10`` stall 44100 cycles.
    """

    name = "all_miss"

    def __init__(self, penalty: int) -> None:
        self.penalty = penalty

    def access(self, t: int, op: int, row: int, region: int) -> int:
        return self.penalty


class PageFirstTouch(MemoryModel):
    """The first access to each page stalls ``penalty`` cycles; later ones are free.

    A page is ``page_rows`` consecutive flat rows. This is a memory that is
    big enough to keep everything but starts empty: the cost is one fetch per
    page touched. Example: rows 0..2047 with ``page_rows = 1024`` are two
    pages, so two stalls.
    """

    name = "page_first_touch"

    def __init__(self, page_rows: int, penalty: int) -> None:
        self.page_rows = page_rows
        self.penalty = penalty
        self._seen: set[int] = set()

    def start(self, trace: Trace) -> None:
        self._seen = set()

    def access(self, t: int, op: int, row: int, region: int) -> int:
        page = row // self.page_rows
        if page in self._seen:
            return 0
        self._seen.add(page)
        return self.penalty

    def counters(self) -> dict[str, Any]:
        return {"pages_touched": len(self._seen)}


class Banked(MemoryModel):
    """``num_banks`` banks, each busy for ``busy_cycles`` after an access.

    An access to a bank that is still busy waits until it is free. With
    ``busy_cycles = 1`` a bank serves one access per cycle and nothing stalls
    unless one word hits the same bank twice.

    The row-to-bank mapping is the option under study:

    - ``"interleave"`` (low-order): ``bank = row % num_banks``. Neighbouring
      rows are in different banks.
    - ``"blocked"``: ``bank = (row // bank_rows) % num_banks``. Each bank holds
      ``bank_rows`` consecutive rows (the hardware's bank = page layout).

    Example with 2 banks, ``busy_cycles = 3``, interleave: accesses to rows 0,
    2, 1 at cycles 10, 11, 12. Row 0 makes bank 0 busy until 13. Row 2 (bank
    0) at cycle 11 waits 2 cycles; it happens at 13 and bank 0 is busy until
    16. Row 1 (bank 1) now arrives at 14 and does not wait.
    """

    name = "banked"

    def __init__(
        self, num_banks: int, busy_cycles: int, mapping: str = "interleave", bank_rows: int = 1024
    ) -> None:
        if mapping not in ("interleave", "blocked"):
            raise ValueError(f"mapping must be 'interleave' or 'blocked', not {mapping!r}")
        self.num_banks = num_banks
        self.busy_cycles = busy_cycles
        self.mapping = mapping
        self.bank_rows = bank_rows
        self._free_at = [0] * num_banks
        self._waits = 0

    def start(self, trace: Trace) -> None:
        self._free_at = [0] * self.num_banks
        self._waits = 0

    def bank_of(self, row: int) -> int:
        if self.mapping == "interleave":
            return row % self.num_banks
        return (row // self.bank_rows) % self.num_banks

    def access(self, t: int, op: int, row: int, region: int) -> int:
        bank = self.bank_of(row)
        stall = max(0, self._free_at[bank] - t)
        self._free_at[bank] = t + stall + self.busy_cycles
        if stall:
            self._waits += 1
        return stall

    def counters(self) -> dict[str, Any]:
        return {"bank_waits": self._waits}


class Cache(MemoryModel):
    """A textbook set-associative cache with LRU replacement.

    Args:
        size_rows: Capacity in rows; ``None`` is an infinite cache (nothing is
            ever evicted), 0 a cache that holds nothing (never hits).
        line_rows: Rows fetched together on a miss (a line).
        assoc: Lines per set. ``size_rows / line_rows / assoc`` sets;
            ``assoc = 0`` means fully associative (one set).
        miss_penalty: Stall cycles of a miss.
        preloaded: The first touch of a line is free, as if a perfect
            prefetch ran before the kernel; a line misses only when it comes
            back after an eviction. A preloaded infinite cache never misses,
            so it replays to the ideal cycle count.
        write_allocate_free: A write miss costs nothing (the row is created
            on chip, nothing is fetched). Otherwise it costs ``miss_penalty``.

    Example: ``size_rows = 4, line_rows = 1``, fully associative, reading rows
    0 1 2 3 0 4 0: five misses (0 1 2 3, then 4 evicts row 1, the least
    recently used) and two hits (the second and third read of row 0).
    """

    name = "cache"

    def __init__(
        self,
        size_rows: int | None,
        line_rows: int = 1,
        assoc: int = 0,
        miss_penalty: int = 10,
        preloaded: bool = False,
        write_allocate_free: bool = False,
    ) -> None:
        self.size_rows = size_rows
        self.line_rows = line_rows
        self.miss_penalty = miss_penalty
        self.preloaded = preloaded
        self.write_allocate_free = write_allocate_free
        lines = None if size_rows is None else size_rows // line_rows
        if lines is None or assoc == 0:
            self.num_sets, self.ways = 1, lines  # fully associative (or infinite)
        else:
            self.num_sets, self.ways = max(1, lines // assoc), assoc
        # One LRU list per set: an OrderedDict keeps lines oldest-first.
        self._sets: list[OrderedDict[int, None]] = []
        self._evicted: set[int] = set()
        self._hits = self._misses = self._evictions = 0

    def start(self, trace: Trace) -> None:
        self._sets = [OrderedDict() for _ in range(self.num_sets)]
        self._evicted = set()
        self._hits = self._misses = self._evictions = 0

    def access(self, t: int, op: int, row: int, region: int) -> int:
        line = row // self.line_rows
        lru = self._sets[line % self.num_sets]
        if line in lru:
            lru.move_to_end(line)  # most recently used
            self._hits += 1
            return 0
        if self.ways == 0:  # a cache that holds nothing
            self._misses += 1
            return 0 if (op == WRITE and self.write_allocate_free) else self.miss_penalty
        if self.ways is not None and len(lru) >= self.ways:
            victim, _ = lru.popitem(last=False)  # evict the least recently used line
            self._evicted.add(victim)
            self._evictions += 1
        lru[line] = None
        # With ``preloaded`` a line costs nothing until it has been evicted
        # once: only bringing it *back* is a real miss.
        if self.preloaded and line not in self._evicted:
            self._hits += 1
            return 0
        self._misses += 1
        return 0 if (op == WRITE and self.write_allocate_free) else self.miss_penalty

    def counters(self) -> dict[str, Any]:
        return {"hits": self._hits, "misses": self._misses, "evictions": self._evictions}
