"""Hand-made traces for the tests."""

from __future__ import annotations

import numpy as np

from ipu_memsim.trace import OTHER, Trace


def make_trace(accesses, regions=(), ideal_cycles=None) -> Trace:
    """A trace from ``(cycle, op, row)`` tuples.

    ``regions`` is ``(name, role, base_row, rows)`` tuples; each access gets
    the region that holds its row.
    """
    region_list = [
        {"name": n, "role": role, "base_row": base, "rows": rows} for n, role, base, rows in regions
    ]
    index = []
    for _, _, row in accesses:
        found = OTHER
        for i, r in enumerate(region_list):
            if r["base_row"] <= row < r["base_row"] + r["rows"]:
                found = i
        index.append(found)
    cycles = [a[0] for a in accesses]
    header = {
        "kernel": "handmade",
        "case": "test",
        "format": "fp32",
        "row_bytes": 512,
        "ideal_cycles": ideal_cycles if ideal_cycles is not None else max(cycles, default=-1) + 1,
        "regions": region_list,
        "stats": {"xmem_reads": 0, "xmem_writes": 0},
        "stats_match": True,
    }
    return Trace(
        header,
        np.asarray(cycles, dtype=np.int64),
        np.asarray([a[1] for a in accesses], dtype=np.int8),
        np.asarray(index, dtype=np.int16),
        np.asarray([a[2] for a in accesses], dtype=np.int64),
        np.full(len(accesses), 512, dtype=np.int32),
    )
