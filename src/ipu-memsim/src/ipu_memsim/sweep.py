"""Sweeps: replay one trace under many memory configurations.

:func:`run_sweeps` runs three sweeps on a trace and writes, for each, the
points to one shared CSV and a line chart (SVG):

1. **Bank count** (``cache_unit``): 3 to 64 banks of 1024 rows.
2. **XMEM size** (0.5, 1, 2, 4, 8 MB), two ways: more banks of 1024 rows, or
   16 banks of more rows. Plus the toy ``cache`` of the same size.
3. **Miss penalty**: for ``cache_unit`` the penalty is the time to fill a
   bank, set by the DRAM bandwidth; for the toy models it is a number.

Sweeps 1 and 2 are drawn for several DRAM speeds, because how much memory
matters depends on whether the DMA can keep up with the IPU.

**Everything here is a TOY-MODEL result** until ``docs/questions.md`` is
answered: the DRAM speeds are placeholders (Q1), each kernel is replayed
alone (Q2), the bank allocation is the tool's guess (Q4).
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ipu_memsim.cache_unit import ROW_BYTES, CacheUnit
from ipu_memsim.dram import FixedDram
from ipu_memsim.models import AllMiss, Cache, MemoryModel, PageFirstTouch
from ipu_memsim.replay import replay
from ipu_memsim.svgplot import Series, line_chart
from ipu_memsim.trace import Trace

__all__ = ["DRAM_SPEEDS", "SweepPoint", "run_sweeps", "sweep"]

MB = 2**20
#: DRAM speeds the bank and size sweeps are drawn for: rows (128 bytes) the
#: DMA moves per IPU cycle. PLACEHOLDERS, not measurements (Q1).
DRAM_SPEEDS = (1.0, 0.1, 0.02)
DRAM_LATENCY = 100
SIZES_MB = (0.5, 1, 2, 4, 8)
BANK_COUNTS = (4, 6, 8, 12, 16, 24, 32, 64)
PENALTIES = (1, 10, 100, 1000, 10000)
BANDWIDTHS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


@dataclass(frozen=True)
class SweepPoint:
    """One replay of a sweep.

    Attributes:
        sweep: The sweep's name (``bank_count``).
        series: The line it belongs to (``cache_unit, DRAM 0.1 rows/cycle``).
        x: The swept value.
        xmem_bytes: XMEM size of this configuration, where it has one (else 0).
        ideal_cycles, stall_cycles, total_cycles: The replay's result.
    """

    sweep: str
    series: str
    x: float
    xmem_bytes: int
    ideal_cycles: int
    stall_cycles: int
    total_cycles: int


def sweep(
    trace: Trace,
    name: str,
    series: str,
    values: Sequence[float],
    build: Callable[[float], tuple[MemoryModel, int]],
) -> list[SweepPoint]:
    """Replay ``trace`` once per value; ``build(value)`` gives the model and
    its XMEM size in bytes. A configuration the model refuses (too few banks
    for the kernel's tables) is left out."""
    points = []
    for value in values:
        try:
            model, size = build(value)
            result = replay(trace, model)
        except ValueError:
            continue
        points.append(
            SweepPoint(
                name,
                series,
                value,
                size,
                result.ideal_cycles,
                result.stall_cycles,
                result.total_cycles,
            )
        )
    return points


def _unit(num_banks: int, bank_rows: int, rows_per_cycle: float) -> tuple[MemoryModel, int]:
    model = CacheUnit(num_banks, bank_rows, FixedDram(DRAM_LATENCY, rows_per_cycle))
    return model, num_banks * bank_rows * ROW_BYTES


def _chart(
    points: Sequence[SweepPoint], *, title: str, x_label: str, log_x: bool, kernel: str
) -> str:
    names = list(dict.fromkeys(p.series for p in points))  # in first-seen order
    series = [Series(n, [(p.x, p.total_cycles) for p in points if p.series == n]) for n in names]
    ideal = points[0].ideal_cycles
    return line_chart(
        series,
        title=title,
        subtitle=f"TOY MODEL. {kernel}, ideal = {ideal:,} cycles.",
        x_label=x_label,
        y_label="estimated runtime (cycles)",
        log_x=log_x,
    )


def run_sweeps(trace: Trace, out: Path) -> list[Path]:
    """Run the three sweeps on ``trace``; write ``sweeps.csv`` and four SVG
    charts into ``out``. Returns the files written."""
    started = time.perf_counter()
    kernel = f"{trace.header['kernel']} ({trace.header['case']})"
    if len(kernel) > 70:
        kernel = kernel[:67] + "..."
    groups: dict[str, list[SweepPoint]] = {}

    groups["bank_count"] = [
        p
        for speed in DRAM_SPEEDS
        for p in sweep(
            trace, "bank_count", f"DRAM {speed:g} rows/cycle", BANK_COUNTS,
            lambda n, speed=speed: _unit(int(n), 1024, speed),
        )
    ]  # fmt: skip
    # XMEM size, at the middle DRAM speed: three ways to spend the bytes.
    speed = DRAM_SPEEDS[1]
    groups["xmem_size"] = (
        sweep(trace, "xmem_size", "cache_unit, banks of 1024 rows", SIZES_MB,
              lambda mb: _unit(int(mb * MB) // (1024 * ROW_BYTES), 1024, speed))
        + sweep(trace, "xmem_size", "cache_unit, 16 banks", SIZES_MB,
                lambda mb: _unit(16, int(mb * MB) // (16 * ROW_BYTES), speed))
        + sweep(trace, "xmem_size", "toy cache, 1-row lines, penalty 110", SIZES_MB,
                lambda mb: (Cache(int(mb * MB) // ROW_BYTES, miss_penalty=110,
                                  write_allocate_free=True), int(mb * MB)))
    )  # fmt: skip
    groups["miss_penalty_toy"] = (
        sweep(trace, "miss_penalty_toy", "all_miss", PENALTIES, lambda p: (AllMiss(int(p)), 0))
        + sweep(trace, "miss_penalty_toy", "toy cache 2 MB, 1-row lines", PENALTIES,
                lambda p: (Cache(2 * MB // ROW_BYTES, miss_penalty=int(p)), 2 * MB))
        + sweep(trace, "miss_penalty_toy", "page_first_touch, 1024-row pages", PENALTIES,
                lambda p: (PageFirstTouch(1024, int(p)), 0))
    )  # fmt: skip
    groups["dram_bandwidth"] = [
        p
        for banks in (4, 8, 16)
        for p in sweep(
            trace, "dram_bandwidth", f"cache_unit, {banks} banks", BANDWIDTHS,
            lambda speed, banks=banks: _unit(banks, 1024, speed),
        )
    ]  # fmt: skip

    out.mkdir(parents=True, exist_ok=True)
    written = [out / "sweeps.csv"]
    with open(written[0], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sweep", "series", "x", "xmem_bytes", "ideal_cycles", "stall_cycles",
                         "total_cycles", "slowdown"])  # fmt: skip
        for points in groups.values():
            for p in points:
                writer.writerow([p.sweep, p.series, f"{p.x:g}", p.xmem_bytes, p.ideal_cycles,
                                 p.stall_cycles, p.total_cycles,
                                 f"{p.total_cycles / p.ideal_cycles:.4f}"])  # fmt: skip
    charts = {
        "bank_count": ("Runtime against bank count (1024-row banks)", "banks in XMEM", True),
        "xmem_size": (f"Runtime against XMEM size (DRAM {speed:g} rows/cycle)", "XMEM size (MB)", True),
        "miss_penalty_toy": ("Runtime against miss penalty (toy models)", "miss penalty (cycles)", True),
        "dram_bandwidth": ("Runtime against DRAM bandwidth", "DRAM bandwidth (rows per cycle)", True),
    }  # fmt: skip
    for name, (title, x_label, log_x) in charts.items():
        if groups[name]:
            path = out / f"{name}.svg"
            path.write_text(
                _chart(groups[name], title=title, x_label=x_label, log_x=log_x, kernel=kernel)
            )
            written.append(path)
    replays = sum(len(points) for points in groups.values())
    (out / "sweep_timing.txt").write_text(
        f"{replays} replays of {len(trace)} accesses in {time.perf_counter() - started:.1f} s\n"
    )
    written.append(out / "sweep_timing.txt")
    return written
