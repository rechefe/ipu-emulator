"""The off-chip side: how long the DMA takes to move rows to or from DRAM/HBM.

:class:`DramBackend` is the interface the ``cache_unit`` model calls. Two
backends:

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

__all__ = ["DramBackend", "FixedDram", "TableDram", "make_dram"]


class DramBackend:
    """How long one DMA transfer takes."""

    def transfer_cycles(self, rows: int, write: bool) -> int:
        """Cycles to move ``rows`` rows in one transfer.

        Args:
            rows: Rows moved (a bank fill moves up to one bank's rows).
            write: XMEM to DRAM (a drain) when true, else DRAM to XMEM (a fill).
        """
        raise NotImplementedError


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

    ``kind = "fixed"`` (default): ``latency_cycles``, ``rows_per_cycle``.
    ``kind = "table"``: ``fill = {"1024" = 2148, ...}`` and optionally ``drain``.
    """
    config = dict(config or {})
    kind = config.pop("kind", "fixed")
    if kind == "fixed":
        return FixedDram(**config)  # type: ignore[arg-type]
    if kind == "table":
        fill = {int(k): int(v) for k, v in dict(config["fill"]).items()}  # type: ignore[call-overload]
        drain = config.get("drain")
        return TableDram(fill, {int(k): int(v) for k, v in dict(drain).items()} if drain else None)  # type: ignore[call-overload]
    raise ValueError(f"unknown [dram] kind {kind!r} (fixed, table)")
