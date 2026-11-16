"""The ``cache_unit`` model: XMEM as the cache-unit specification describes it.

Unlike the toy models of ``ipu_memsim.models`` this one follows
``docs/content/specs/cache-unit.md`` and
``docs/content/requirements/memory-access.md``. What it takes from them
(CONFIRMED, see ``docs/memory_sketch.md`` for the line references):

- XMEM is ``num_banks`` banks of ``bank_rows`` rows; bank = page.
- Each table (ASSUMED: one ``kernel.toml`` region = one table) owns an ordered
  list of banks. Page ``p`` of a table lives in the table's bank number
  ``p % (banks of the table)``: a direct-mapped cache of pages.
- An access hits when that bank holds page ``p``; otherwise the IPU stalls
  until it does.
- A read page comes from DRAM, and is usable only when the *whole* page has
  arrived (CU-21). A write to a new page claims a free bank at no DRAM cost
  (CU-22).
- ``jump_back``: an access at row ``>= jump_back`` inside page ``p > 0``
  gives page ``p - 1`` back. A clean bank is free at once; a written (dirty)
  bank is first drained to DRAM by the DMA.
- The DMA works in the background, also while the IPU computes or stalls.
  That is why this model keeps absolute time: ``access`` is told the real
  time ``t`` of each access, and every bank remembers when its data is ready.

What it has to assume (ASSUMED, each a constructor argument or a ``[dram]``
setting; the questions are in ``docs/questions.md``):

- how long a DMA transfer takes (``dram``, Q1) and that there is one DMA
  serving transfers first come first served (``dma_channels``, Q10);
- how many banks each table gets (``bank_allocation``, Q4);
- each table's ``jump_back`` (``jump_back``, Q8);
- that the DMA prefetches the next page of an input table when its bank is
  free (``prefetch``), and whether banks are already filled when the kernel
  starts (``prefetch_before_start``, Q3);
- that a kernel which needs a page whose bank holds another page of the same
  table evicts that page (drained first if dirty) and fetches the new one.
  The specification has no such eviction (CU-20): there the table must be set
  up so this never happens. The model counts these as ``conflicts`` so a bad
  setup is visible instead of silent (Q5);
- that the DMA moves only the rows a table really has in a page, not always
  a full bank (``fill_whole_bank``, Q9).

Worked example. A 2048-row input table read once, front to back, one row per
cycle, with 2 banks, ``jump_back = 0``, and a DRAM that needs 1100 cycles per
full page. Cycle 0 reads page 0: miss, the DMA fetches page 0 (ready at 1100),
the IPU stalls 1100 cycles. While reading page 0 the model prefetches page 1
into the second bank (ready at 2200). The IPU reaches page 1 at time
1100 + 1024 = 2124 and has to wait 76 more cycles. Total stall 1176 cycles
for 2048 cycles of work. With ``prefetch_before_start`` both pages are ready
at time 0 and nothing stalls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ipu_memsim.dram import DramBackend, FixedDram
from ipu_memsim.models import MemoryModel
from ipu_memsim.trace import OTHER, WRITE, Trace

__all__ = ["CacheUnit", "DmaTransfer"]

#: Hardware bytes per row (1024 bits), for size reports.
ROW_BYTES = 128


@dataclass
class DmaTransfer:
    """One DMA transfer the model scheduled (kept for the DRAM trace export).

    Attributes:
        begin, end: The cycles it occupies the DMA.
        table: The table's name.
        page: The page moved.
        rows: Rows moved.
        write: XMEM to DRAM (drain) when true, else DRAM to XMEM (fill).
    """

    begin: int
    end: int
    table: str
    page: int
    rows: int
    write: bool


@dataclass
class _Bank:
    """One XMEM bank as its table sees it."""

    page: int | None = None  # the page it holds; None = free
    ready: int = 0  # when its data is usable (a fill may still be running)
    dirty: bool = False  # the IPU wrote to it since it was claimed
    free_at: int = 0  # when a background drain of its old page ends


@dataclass
class _Table:
    """One table: a region of the kernel and the banks it owns."""

    name: str
    role: str
    base_row: int
    rows: int
    pages: int
    jump_back: int
    banks: list[_Bank] = field(default_factory=list)
    in_dram: set[int] = field(default_factory=set)  # pages whose data is in DRAM
    fetched: set[int] = field(default_factory=set)  # pages fetched at least once
    stats: dict[str, int] = field(default_factory=dict)


_STAT_NAMES = (
    "hits", "waits", "misses", "fetches", "refetches", "claims", "conflicts", "drains",
    "prefetches",
)  # fmt: skip


class CacheUnit(MemoryModel):
    """XMEM banks, tables, ``jump_back`` and a background DMA (see the module).

    Args:
        num_banks: Banks in XMEM (hardware: 16).
        bank_rows: Rows per bank = page size (hardware: 1024).
        dram: The off-chip timing model.
        bank_allocation: ``"proportional"`` (banks shared by table size, at
            least one each) or ``{table name: banks}``.
        jump_back: ``"auto"`` (per table, the largest distance the kernel
            ever reads or writes behind its furthest row), an integer for
            every table, or ``"never"`` (no bank is given back early).
        prefetch: The DMA fills the next page of an input table as soon as
            its bank is free.
        prefetch_before_start: Input pages that fit their banks are already
            in XMEM at cycle 0.
        dma_channels: Transfers that can run at the same time.
        fill_whole_bank: A transfer always moves ``bank_rows`` rows.
    """

    name = "cache_unit"

    def __init__(  # noqa: PLR0913 (each argument is one config key)
        self,
        num_banks: int = 16,
        bank_rows: int = 1024,
        dram: DramBackend | None = None,
        bank_allocation: str | Mapping[str, int] = "proportional",
        jump_back: str | int = "auto",
        prefetch: bool = True,
        prefetch_before_start: bool = False,
        dma_channels: int = 1,
        fill_whole_bank: bool = False,
    ) -> None:
        self.num_banks = num_banks
        self.bank_rows = bank_rows
        self.dram = dram or FixedDram()
        self.bank_allocation = bank_allocation
        self.jump_back = jump_back
        self.prefetch = prefetch
        self.prefetch_before_start = prefetch_before_start
        self.dma_channels = dma_channels
        self.fill_whole_bank = fill_whole_bank
        self.transfers: list[DmaTransfer] = []
        self._tables: dict[int, _Table] = {}
        self._dma_free: list[int] = [0] * dma_channels
        self._tail_drain = 0
        self._other = 0

    # -- setup ---------------------------------------------------------------

    def start(self, trace: Trace) -> None:
        self.transfers = []
        self._dma_free = [0] * self.dma_channels
        self._tail_drain = 0
        self._other = 0
        self._tables = {}
        # Only regions the program really touches become tables.
        for index, region in enumerate(trace.header["regions"]):
            mine = trace.region == index
            if not mine.any():
                continue
            offsets = trace.row[mine] - region["base_row"]
            self._tables[index] = _Table(
                name=region["name"],
                role=region["role"],
                base_row=region["base_row"],
                rows=region["rows"],
                pages=-(-region["rows"] // self.bank_rows),  # ceiling division
                # A scratch table stays resident for the whole kernel (spec 5.1).
                jump_back=(
                    self.bank_rows if region["role"] == "scratch" else self._jump_back_for(offsets)
                ),
                stats=dict.fromkeys(_STAT_NAMES, 0),
            )
        for table, count in zip(self._tables.values(), self._allocate(), strict=True):
            table.banks = [_Bank() for _ in range(count)]
            if table.role == "input":
                table.in_dram = set(range(table.pages))  # inputs start in DRAM
                if self.prefetch_before_start:
                    for page in range(min(count, table.pages)):
                        table.banks[page].page = page  # ready at cycle 0
                        table.fetched.add(page)

    def _jump_back_for(self, offsets: np.ndarray) -> int:
        """The table's ``jump_back`` in rows (``bank_rows`` or more: never)."""
        if self.jump_back == "never":
            return self.bank_rows
        if self.jump_back == "auto":
            # How far behind its furthest row so far does the kernel ever go?
            furthest = np.maximum.accumulate(offsets)
            return int((furthest - offsets).max())
        return int(self.jump_back)

    def _allocate(self) -> list[int]:
        """Banks per table, in table order."""
        tables = list(self._tables.values())
        if not tables:
            return []
        if isinstance(self.bank_allocation, Mapping):
            counts = [int(self.bank_allocation[t.name]) for t in tables]
        else:
            if self.num_banks < len(tables):
                raise ValueError(
                    f"{len(tables)} tables need at least {len(tables)} banks, "
                    f"XMEM has {self.num_banks}"
                )
            counts = [1] * len(tables)
            # Hand out the remaining banks one at a time to the table that is
            # furthest from holding all its pages.
            for _ in range(self.num_banks - len(tables)):
                short = [
                    t.pages / c if t.pages > c else 0.0 for t, c in zip(tables, counts, strict=True)
                ]
                if max(short) == 0.0:
                    break  # every table is fully resident
                counts[short.index(max(short))] += 1
        if sum(counts) > self.num_banks or min(counts, default=1) < 1:
            raise ValueError(f"bank allocation {counts} does not fit {self.num_banks} banks")
        return counts

    # -- the DMA -------------------------------------------------------------

    def _rows_of(self, table: _Table, page: int) -> int:
        if self.fill_whole_bank:
            return self.bank_rows
        return min(self.bank_rows, table.rows - page * self.bank_rows)

    def _dma(self, earliest: int, table: _Table, page: int, write: bool) -> int:
        """Schedule one transfer; return the cycle it completes."""
        channel = self._dma_free.index(min(self._dma_free))
        begin = max(earliest, self._dma_free[channel])
        rows = self._rows_of(table, page)
        end = begin + self.dram.transfer_cycles(rows, write)
        self._dma_free[channel] = end
        self.transfers.append(DmaTransfer(begin, end, table.name, page, rows, write))
        return end

    # -- one access ----------------------------------------------------------

    def access(self, t: int, op: int, row: int, region: int) -> int:
        if region == OTHER:
            self._other += 1  # outside every table: not modelled, never stalls
            return 0
        table = self._tables[region]
        stats = table.stats
        page, row_in_page = divmod(row - table.base_row, self.bank_rows)
        bank = table.banks[page % len(table.banks)]

        if bank.page == page:
            ready = bank.ready  # a hit; the fill may still be on its way
            stats["waits" if ready > t else "hits"] += 1
        else:
            stats["misses"] += 1
            ready = t
            if bank.page is not None:
                # The bank holds another page of this table: evict it.
                stats["conflicts"] += 1
                if bank.dirty:
                    ready = self._dma(max(t, bank.ready), table, bank.page, write=True)
                    table.in_dram.add(bank.page)
                    stats["drains"] += 1
            ready = max(ready, bank.free_at)
            if page in table.in_dram:
                ready = self._dma(ready, table, page, write=False)
                stats["refetches" if page in table.fetched else "fetches"] += 1
                table.fetched.add(page)
            else:
                stats["claims"] += 1  # a new page: nothing to fetch
            bank.page, bank.ready, bank.dirty, bank.free_at = page, ready, False, 0

        stall = max(0, ready - t)
        if op == WRITE:
            bank.dirty = True
        now = t + stall

        if page > 0 and row_in_page >= table.jump_back:
            self._release(table, page - 1, now)
        if self.prefetch and page + 1 < table.pages and page + 1 in table.in_dram:
            ahead = table.banks[(page + 1) % len(table.banks)]
            if ahead.page is None:
                ahead.ready = self._dma(max(now, ahead.free_at), table, page + 1, write=False)
                ahead.page, ahead.dirty, ahead.free_at = page + 1, False, 0
                table.fetched.add(page + 1)
                stats["prefetches"] += 1
        return stall

    def _release(self, table: _Table, page: int, now: int) -> None:
        """Give ``page``'s bank back (the jump-back point of that page)."""
        bank = table.banks[page % len(table.banks)]
        if bank.page != page:
            return
        if bank.dirty:
            bank.free_at = self._dma(max(now, bank.ready), table, page, write=True)
            table.in_dram.add(page)
            table.stats["drains"] += 1
        bank.page, bank.dirty = None, False

    # -- the end of the kernel -----------------------------------------------

    def finish(self, t: int) -> None:
        """Drain what the kernel left in written banks (CU-17). The kernel is
        done at ``t`` and does not wait for this (IPU-EXEC-18); the time is
        reported as ``tail_drain_cycles``."""
        for table in self._tables.values():
            if table.role == "scratch":
                continue
            for bank in table.banks:
                if bank.page is not None and bank.dirty:
                    self._dma(max(t, bank.ready), table, bank.page, write=True)
                    table.stats["drains"] += 1
                    bank.dirty = False
        self._tail_drain = max(0, max(self._dma_free) - t)

    def counters(self) -> dict[str, Any]:
        totals = {name: sum(t.stats[name] for t in self._tables.values()) for name in _STAT_NAMES}
        return {
            **totals,
            "xmem_bytes": self.num_banks * self.bank_rows * ROW_BYTES,
            "banks_used": sum(len(t.banks) for t in self._tables.values()),
            "dma_transfers": len(self.transfers),
            "dma_busy_cycles": sum(x.end - x.begin for x in self.transfers),
            "tail_drain_cycles": self._tail_drain,
            "untracked_accesses": self._other,
            "tables": {
                t.name: {
                    "role": t.role,
                    "rows": t.rows,
                    "pages": t.pages,
                    "banks": len(t.banks),
                    "jump_back": t.jump_back,
                    **t.stats,
                }
                for t in self._tables.values()
            },
        }
