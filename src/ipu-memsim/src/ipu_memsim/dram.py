"""The off-chip side: how long the DMA takes to move rows to or from DRAM/HBM.

:class:`DramBackend` is the interface the ``cache_unit`` model calls. Three
backends:

- :class:`BankFillDram`: the one number the architects can give directly,
  the cycles to fill one 1024-row bank. This is the swept parameter of the
  thesis study.
- :class:`FixedDram` (TOY MODEL): a fixed latency plus a bandwidth.
- :class:`TableDram`: latencies measured elsewhere (for example by DRAMsim3
  or Ramulator 2 on the transfer log this tool exports, see
  ``ipu_memsim.dram_trace``), looked up by transfer size.

Units: time in IPU clock cycles, data in XMEM rows (one row = 128 bytes on
the hardware).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = [
    "ASSUMED_CLOCK_NS",
    "DEFAULT_FILL_CYCLES",
    "HBM2_DRAMSIM3_FILL_CYCLES",
    "BankFillDram",
    "DramBackend",
    "FixedDram",
    "TableDram",
    "make_dram",
]

#: The IPU clock period used to convert nanoseconds to cycles. ASSUMED: the
#: emulator defines no clock; open question OQ-16 speaks of meeting 1 GHz.
ASSUMED_CLOCK_NS = 1.0

#: One labelled anchor point, not the truth: DRAMsim3's HBM2_8Gb_x128
#: configuration, one channel, consecutive addresses, fills a 1024-row bank in
#: 2111 DRAM cycles of 1 ns (``memsim dram-measure``, STATUS.md).
HBM2_DRAMSIM3_FILL_CYCLES = 2111

#: The fill time used when a config gives none: one row per cycle. A
#: PLACEHOLDER (UNKNOWN, question Q1), chosen round so nobody mistakes it for
#: a measurement.
DEFAULT_FILL_CYCLES = 1024

#: Rows of the bank that ``fill_cycles_per_bank`` speaks of.
REFERENCE_BANK_ROWS = 1024


class DramBackend:
    """How long one DMA transfer takes."""

    def transfer_cycles(self, rows: int, write: bool) -> int:
        """Cycles to move ``rows`` rows in one transfer.

        Args:
            rows: Rows moved (a bank fill moves up to one bank's rows).
            write: XMEM to DRAM (a drain) when true, else DRAM to XMEM (a fill).
        """
        raise NotImplementedError


class BankFillDram(DramBackend):
    """Transfer time from one number: the cycles to fill a 1024-row bank.

    A transfer of fewer or more rows takes proportionally less or more
    (ASSUMED: no fixed start-up cost; question Q9), rounded up. Draining a
    bank takes ``drain_cycles_per_bank``, the same as a fill unless given.

    The time can be given in nanoseconds instead (``fill_ns_per_bank``); it is
    then divided by the chip clock period ``clock_ns``.

    Example: ``fill_cycles_per_bank = 2111``. A full bank takes 2111 cycles, a
    64-row table ``ceil(2111 * 64 / 1024) = 132`` cycles. The same fill given
    as 2111 ns on a 0.5 ns clock is 4222 cycles.
    """

    def __init__(
        self,
        fill_cycles_per_bank: float | None = None,
        drain_cycles_per_bank: float | None = None,
        fill_ns_per_bank: float | None = None,
        clock_ns: float = ASSUMED_CLOCK_NS,
    ) -> None:
        if clock_ns <= 0:
            raise ValueError("clock_ns must be positive")
        if fill_ns_per_bank is not None:
            if fill_cycles_per_bank is not None:
                raise ValueError("give fill_cycles_per_bank or fill_ns_per_bank, not both")
            fill_cycles_per_bank = fill_ns_per_bank / clock_ns
        if fill_cycles_per_bank is None:
            fill_cycles_per_bank = DEFAULT_FILL_CYCLES
        if fill_cycles_per_bank < 0:
            raise ValueError("the fill time cannot be negative")
        self.fill_cycles_per_bank = fill_cycles_per_bank
        self.drain_cycles_per_bank = (
            fill_cycles_per_bank if drain_cycles_per_bank is None else drain_cycles_per_bank
        )

    def transfer_cycles(self, rows: int, write: bool) -> int:
        per_bank = self.drain_cycles_per_bank if write else self.fill_cycles_per_bank
        return math.ceil(per_bank * rows / REFERENCE_BANK_ROWS)


class FixedDram(DramBackend):
    """TOY MODEL: ``latency_cycles + rows / rows_per_cycle``, rounded up.

    Example: ``latency_cycles = 100`` and ``rows_per_cycle = 0.5`` (64 bytes
    per cycle). Filling a whole 1024-row bank takes ``100 + 1024 / 0.5 = 2148``
    cycles.

    The defaults are placeholders, not measurements (question Q1 in
    ``docs/questions.md``).
    """

    def __init__(self, latency_cycles: int = 100, rows_per_cycle: float = 1.0) -> None:
        if rows_per_cycle <= 0:
            raise ValueError("rows_per_cycle must be positive")
        self.latency_cycles = latency_cycles
        self.rows_per_cycle = rows_per_cycle

    def transfer_cycles(self, rows: int, write: bool) -> int:
        return self.latency_cycles + math.ceil(rows / self.rows_per_cycle)


class TableDram(DramBackend):
    """Transfer times from a table ``{rows: cycles}``, for results that come
    from an external DRAM simulator.

    A size not in the table is interpolated linearly between its neighbours
    (and extrapolated from the largest entry by its cycles-per-row).
    """

    def __init__(self, fill: Mapping[int, int], drain: Mapping[int, int] | None = None) -> None:
        if not fill:
            raise ValueError("the fill table is empty")
        self._fill = sorted(fill.items())
        self._drain = sorted((drain or fill).items())

    @staticmethod
    def _lookup(table: list[tuple[int, int]], rows: int) -> int:
        previous = (0, 0)
        for size, cycles in table:
            if rows <= size:
                span = size - previous[0]
                part = (rows - previous[0]) / span if span else 1.0
                return math.ceil(previous[1] + part * (cycles - previous[1]))
            previous = (size, cycles)
        size, cycles = table[-1]
        return math.ceil(cycles * rows / size)

    def transfer_cycles(self, rows: int, write: bool) -> int:
        return self._lookup(self._drain if write else self._fill, rows)


def make_dram(config: Mapping[str, object] | None) -> DramBackend:
    """A backend from the ``[dram]`` table of a config file.

    ``kind = "bank_fill"`` (default): ``fill_cycles_per_bank`` (or ``fill_ns_per_bank``
    with ``clock_ns``), optionally ``drain_cycles_per_bank``.
    ``kind = "fixed"``: ``latency_cycles``, ``rows_per_cycle``.
    ``kind = "table"``: ``fill = {"1024" = 2148, ...}`` and optionally ``drain``.
    """
    config = dict(config or {})
    kind = config.pop("kind", "bank_fill")
    if kind == "bank_fill":
        return BankFillDram(**config)  # type: ignore[arg-type]
    if kind == "fixed":
        return FixedDram(**config)  # type: ignore[arg-type]
    if kind == "table":
        fill = {int(k): int(v) for k, v in dict(config["fill"]).items()}  # type: ignore[call-overload]
        drain = config.get("drain")
        return TableDram(fill, {int(k): int(v) for k, v in dict(drain).items()} if drain else None)  # type: ignore[call-overload]
    raise ValueError(f"unknown [dram] kind {kind!r} (bank_fill, fixed, table)")
