"""Replay a trace through a memory model and add up the stall cycles.

The one timing assumption of the whole tool is here (ASSUMED, see STATUS.md):

    **Stalls serialize.**  estimated runtime = ideal cycles + sum of stalls.

Why this is reasonable: the cache agent is the IPU's only source of
backpressure and it stalls CTRL only (IPU-ARCH-14); a stalled word has changed
nothing (IPU-ARCH-17). So a stall delays every later word by the same number
of cycles and cannot change which rows the program touches or in which order.
The replay therefore walks the recorded accesses in order and keeps one
number, ``shift``: the stall cycles accumulated so far. An access the emulator
made at cycle ``c`` reaches the memory model at time ``c + shift``.

A word with a load and a store has two accesses in the same cycle. They are
given to the model one after the other (load first, as recorded); the second
sees the time after the first one's stall. For a fixed-penalty model the two
stalls therefore add up.

Example: a trace of 1000 cycles with 3 accesses that stall 50 cycles each
gives 1150 cycles. The third access, recorded at cycle 700, reaches the model
at time 800.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ipu_memsim.models import MemoryModel
from ipu_memsim.trace import OTHER, READ, Trace

__all__ = ["RegionStats", "ReplayResult", "replay"]


@dataclass
class RegionStats:
    """What one region saw in a replay.

    Attributes:
        reads, writes: Accesses of each kind.
        stalled: Accesses that stalled at all.
        stall_cycles: Their stall cycles, summed.
    """

    reads: int = 0
    writes: int = 0
    stalled: int = 0
    stall_cycles: int = 0


@dataclass
class ReplayResult:
    """The outcome of one replay.

    Attributes:
        model: The model's name.
        ideal_cycles: The emulator's cycle count (no memory stalls).
        stall_cycles: Stall cycles the model reported, summed.
        total_cycles: ``ideal_cycles + stall_cycles``.
        accesses: Trace entries replayed.
        regions: Per-region statistics, by region name.
        counters: The model's own counters (hits, misses, fetches, ...).
    """

    model: str
    ideal_cycles: int
    stall_cycles: int
    accesses: int
    regions: dict[str, RegionStats] = field(default_factory=dict)
    counters: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cycles(self) -> int:
        return self.ideal_cycles + self.stall_cycles

    @property
    def slowdown(self) -> float:
        """``total / ideal`` (1.0 = no stalls)."""
        return self.total_cycles / self.ideal_cycles if self.ideal_cycles else 1.0


def replay(trace: Trace, model: MemoryModel) -> ReplayResult:
    """Run ``trace`` through ``model``.

    Deterministic: the same trace and the same model configuration give the
    same result (no model uses randomness or wall-clock time).
    """
    model.start(trace)
    names = [*trace.region_names, "other"]
    per_region = [RegionStats() for _ in names]  # the last one is OTHER's
    access = model.access  # bound once: this loop is the tool's hot path
    shift = 0
    # Plain Python lists are faster to walk than numpy arrays one item at a time.
    for cycle, op, row, region in zip(
        trace.cycle.tolist(),
        trace.op.tolist(),
        trace.row.tolist(),
        trace.region.tolist(),
        strict=True,
    ):
        stall = access(cycle + shift, op, row, region)
        stats = per_region[region]  # OTHER is -1: the last entry
        if op == READ:
            stats.reads += 1
        else:
            stats.writes += 1
        if stall:
            shift += stall
            stats.stalled += 1
            stats.stall_cycles += stall
    assert OTHER == -1  # the indexing above relies on it
    model.finish(trace.ideal_cycles + shift)
    return ReplayResult(
        model=model.name,
        ideal_cycles=trace.ideal_cycles,
        stall_cycles=shift,
        accesses=len(trace),
        regions={
            name: stats
            for name, stats in zip(names, per_region, strict=True)
            if stats.reads or stats.writes
        },
        counters=model.counters(),
    )
