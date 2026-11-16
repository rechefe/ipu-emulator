"""The adapter to an external DRAM simulator (DRAMsim3), which models HBM.

Nothing here is C++ and nothing links against the simulator: DRAMsim3 is run
as a separate program on trace files this module writes, and its statistics
file is read back.

Two uses:

1. :func:`measure_fill_table` asks DRAMsim3 one question, several times: *how
   many DRAM clock cycles does it take to read N consecutive XMEM rows?* The
   answers, converted to IPU cycles, become a ``[dram] kind = "table"``
   backend (:class:`ipu_memsim.dram.TableDram`) for the ``cache_unit`` model.
   This replaces the placeholder numbers of :class:`~ipu_memsim.dram.FixedDram`
   with ones derived from a DRAM timing model.
2. :func:`write_transfer_trace` writes every DMA transfer of a replay as a
   DRAMsim3 trace, for a full run of the DRAM simulator on the real traffic.

DRAMsim3's trace format is one transaction per line, ``<hex address>
<READ|WRITE> <cycle>``; a transaction moves one burst, ``bus_width * BL / 8``
bytes (64 bytes for the HBM2 configuration shipped with DRAMsim3), so one
128-byte XMEM row is two transactions.

ASSUMED (question Q1): the DMA reads a bank as consecutive addresses, as fast
as the DRAM accepts them, and the IPU clock period is ``ipu_clock_ns``.

Getting DRAMsim3 (not part of this repository)::

    git clone https://github.com/umd-memsys/DRAMsim3.git && cd DRAMsim3
    make dramsim3main.out          # its plain Makefile; no cmake needed
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ipu_memsim.cache_unit import ROW_BYTES, DmaTransfer

__all__ = [
    "burst_bytes",
    "clock_ns",
    "dram_cycles_to_read",
    "measure_fill_table",
    "write_fill_trace",
    "write_transfer_trace",
]


def _ini_number(ini: Path, key: str) -> float:
    match = re.search(rf"^{key}\s*=\s*([0-9.]+)", ini.read_text(), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{ini}: no '{key} = ' line")
    return float(match.group(1))


def burst_bytes(ini: Path) -> int:
    """Bytes one DRAMsim3 transaction moves under configuration ``ini``."""
    return int(_ini_number(ini, "bus_width") * _ini_number(ini, "BL")) // 8


def clock_ns(ini: Path) -> float:
    """The DRAM clock period (``tCK``) of configuration ``ini``, in ns."""
    return _ini_number(ini, "tCK")


def write_fill_trace(
    path: Path, rows: int, burst: int, *, write: bool = False, base: int = 0
) -> int:
    """Write the trace of one transfer of ``rows`` consecutive rows, every
    transaction requested at cycle 0. Returns the number of transactions."""
    count = math.ceil(rows * ROW_BYTES / burst)
    op = "WRITE" if write else "READ"
    path.write_text("".join(f"0x{base + i * burst:X} {op} 0\n" for i in range(count)))
    return count


def write_transfer_trace(
    path: Path, transfers: Sequence[DmaTransfer], burst: int, cycle_scale: float = 1.0
) -> None:
    """Write a replay's DMA transfers (``CacheUnit.transfers``) as one trace.

    Each table gets its own 16 MiB address range in table order; a page is at
    ``page * rows * 128`` inside it. ``cycle_scale`` converts IPU cycles to
    DRAM cycles (IPU clock period / DRAM clock period).
    """
    bases: dict[str, int] = {}
    lines = []
    for transfer in sorted(transfers, key=lambda t: t.begin):
        base = bases.setdefault(transfer.table, len(bases) << 24)
        start = base + transfer.page * 1024 * ROW_BYTES
        op = "WRITE" if transfer.write else "READ"
        cycle = int(transfer.begin * cycle_scale)
        for i in range(math.ceil(transfer.rows * ROW_BYTES / burst)):
            lines.append(f"0x{start + i * burst:X} {op} {cycle}\n")
    path.write_text("".join(lines))


def _done(binary: Path, ini: Path, trace: Path, cycles: int, write: bool, work: Path) -> int:
    """Transactions DRAMsim3 completes of ``trace`` in ``cycles`` DRAM cycles."""
    result = work / "dramsim3.json"
    result.unlink(missing_ok=True)
    # The exit status is not checked: the macOS build of DRAMsim3 aborts while
    # exiting, after it has written its statistics. A run counts when the
    # statistics file is there and parses.
    done = subprocess.run(
        [str(binary), str(ini), "-c", str(cycles), "-t", str(trace), "-o", str(work)],
        check=False, capture_output=True,
    )  # fmt: skip
    if not result.exists():
        raise RuntimeError(f"DRAMsim3 wrote no statistics: {done.stderr.decode()[-300:]}")
    stats = json.loads(result.read_text())
    key = "num_writes_done" if write else "num_reads_done"
    return sum(int(channel[key]) for channel in stats.values())  # one entry per DRAM channel


def dram_cycles_to_read(binary: Path, ini: Path, rows: int, *, write: bool = False) -> int:
    """The fewest DRAM cycles in which DRAMsim3 completes a transfer of
    ``rows`` consecutive rows.

    DRAMsim3 runs for a fixed number of cycles and reports how many
    transactions finished, so the completion time is found by trying cycle
    counts: double until everything finishes, then bisect.
    """
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        count = write_fill_trace(work / "fill.trace", rows, burst_bytes(ini), write=write)
        high = 64
        while _done(binary, ini, work / "fill.trace", high, write, work) < count:
            high *= 2
            if high > 1 << 26:
                raise RuntimeError(f"DRAMsim3 did not finish {count} transactions")
        low = high // 2  # not enough (or the starting point)
        while high - low > 1:
            middle = (low + high) // 2
            if _done(binary, ini, work / "fill.trace", middle, write, work) >= count:
                high = middle
            else:
                low = middle
        return high


def measure_fill_table(
    binary: Path,
    ini: Path,
    sizes: Sequence[int] = (16, 64, 256, 1024, 4096),
    ipu_clock_ns: float = 1.0,
) -> dict[str, dict[str, int]]:
    """Measure fills and drains of each size with DRAMsim3.

    Returns:
        ``{"fill": {rows: IPU cycles}, "drain": {...}}``, the two tables of a
        ``[dram] kind = "table"`` config (keys as strings, as TOML wants).
    """
    scale = clock_ns(ini) / ipu_clock_ns  # DRAM cycles -> IPU cycles
    table: dict[str, dict[str, int]] = {"fill": {}, "drain": {}}
    for rows in sizes:
        table["fill"][str(rows)] = math.ceil(dram_cycles_to_read(binary, ini, rows) * scale)
        table["drain"][str(rows)] = math.ceil(
            dram_cycles_to_read(binary, ini, rows, write=True) * scale
        )
    return table
