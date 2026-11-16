"""Record a kernel's XMEM access trace by running it once on the emulator.

The recorder changes nothing in the emulator. It uses only the reference
model's public API (``ipu_emu.reference.ReferenceModel``):

- ``observe_xmem_reads(callback)`` reports every row the program loads;
- ``step()`` runs one word and returns its commit record, which lists the
  XMEM byte ranges the word wrote and the word's cycle number.

The kernel is prepared exactly as the harness prepares it for a test
(``ipu_harness.runner.stage``: inputs generated and written, registers
programmed), then stepped word by word until it halts. Reads and writes that
happen while staging (the host writing the inputs) are not program accesses
and are not recorded.

A read is one trace entry per load. A write is one entry per row written:
a word's written byte range is cut into rows. In FP32 one store is one row, so
the entries equal the emulator's own ``xmem_reads`` / ``xmem_writes``
statistics; the recorder checks this and stores the result in the header
(``stats_match``). In the 8-bit formats ``ST.AAQ`` writes four rows per store
(open question OQ-35), so there the write entries exceed the statistic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, at_values, expand
from ipu_harness.manifest import KernelManifest
from ipu_harness.registry import default_registry
from ipu_harness.runner import Staged, stage
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_memsim.trace import OTHER, READ, WRITE, Trace

__all__ = ["find_case", "record", "record_staged"]

#: Stop a run that never halts (the emulator's own default limit is 1e6).
MAX_CYCLES = 50_000_000


def find_case(
    manifest: KernelManifest, case: str = "default", values: Mapping[str, object] | None = None
) -> Case:
    """The case ``case`` of a kernel, optionally at other parameter values.

    Args:
        manifest: The kernel.
        case: A case name as the harness lists it (``default``,
            ``sweep[num_channels=96]``).
        values: Parameter values that replace the case's (other shapes).

    Raises:
        KeyError: The kernel has no such case.
    """
    cases = {c.name: c for c in expand(manifest)}
    if case not in cases:
        raise KeyError(f"{manifest.name} has no case {case!r} (cases: {', '.join(cases)})")
    found = cases[case]
    if values:
        label = ",".join(f"{k}={v}" for k, v in values.items())
        found = at_values(found, f"{case}[{label}]", values)
    return found


def _region_index(regions: list[dict[str, Any]], rows: np.ndarray) -> np.ndarray:
    """For each row, the index of the region that holds it, else ``OTHER``."""
    index = np.full(len(rows), OTHER, dtype=np.int16)
    for i, region in enumerate(regions):
        inside = (rows >= region["base_row"]) & (rows < region["base_row"] + region["rows"])
        index[inside] = i
    return index


def record_staged(staged: Staged, header: dict[str, Any]) -> Trace:
    """Run a staged kernel to its HALT and return its trace.

    Args:
        staged: The kernel, loaded and ready to run (``ipu_harness.runner.stage``).
        header: Facts to keep with the trace (kernel name, case, commit).

    Raises:
        RuntimeError: The run faulted, or did not halt in ``MAX_CYCLES``.
    """
    model = staged.model
    row_bytes = FORMATS[staged.plan.format].row_bytes
    cycles: list[int] = []
    ops: list[int] = []
    rows: list[int] = []
    sizes: list[int] = []

    reads_of_word: list[tuple[int, int]] = []
    detach = model.observe_xmem_reads(lambda address, size: reads_of_word.append((address, size)))
    try:
        for _ in range(MAX_CYCLES):
            reads_of_word.clear()
            commit = model.step()
            # Loads first, then stores: the order the cache agent asks for
            # banks in (IPU-MEM-6, "the request for the load goes first").
            for address, size in reads_of_word:
                cycles.append(commit.cycle)
                ops.append(READ)
                rows.append(address // row_bytes)
                sizes.append(size)
            for write in commit.xmem_writes:
                # One entry per row of the written byte range.
                first = write.address // row_bytes
                last = (write.address + len(write.data) - 1) // row_bytes
                for row in range(first, last + 1):
                    cycles.append(commit.cycle)
                    ops.append(WRITE)
                    rows.append(row)
                    sizes.append(row_bytes)
            if commit.outcome == "fault":
                raise RuntimeError(f"the run faulted: {commit.fault}")
            if commit.outcome == "halt":
                break
        else:
            raise RuntimeError(f"no HALT within {MAX_CYCLES} cycles")
    finally:
        detach()

    stats = model.stats()
    regions = [
        {"name": r.name, "role": r.role, "base_row": r.base_row, "rows": r.rows}
        for r in staged.placement.regions.values()
    ]
    row_array = np.asarray(rows, dtype=np.int64)
    op_array = np.asarray(ops, dtype=np.int8)
    reads = int((op_array == READ).sum())
    writes = int((op_array == WRITE).sum())
    full_header = {
        **header,
        "format": staged.plan.format.value,
        "row_bytes": row_bytes,
        "ideal_cycles": stats.cycles,
        "regions": regions,
        "stats": {
            "xmem_reads": stats.xmem_reads,
            "xmem_writes": stats.xmem_writes,
            "mult_active_cycles": stats.mult_active_cycles,
            "acc_active_cycles": stats.acc_active_cycles,
        },
        "stats_match": reads == stats.xmem_reads and writes == stats.xmem_writes,
    }
    return Trace(
        full_header,
        np.asarray(cycles, dtype=np.int64),
        op_array,
        _region_index(regions, row_array),
        row_array,
        np.asarray(sizes, dtype=np.int32),
    )


def record(
    kernel: str,
    case: str = "default",
    values: Mapping[str, object] | None = None,
    fmt: str | None = None,
    *,
    commit: str = "unknown",
    cache: AssemblyCache | None = None,
) -> Trace:
    """Run kernel ``kernel`` once and return its XMEM access trace.

    Args:
        kernel: A registered kernel name (``residual_add``).
        case: One of its cases.
        values: Parameter values over the case's (``{"num_channels": 96}``).
        fmt: The element format (``"fp32"``); ``None``: the case's first.
        commit: The emulator's git commit, kept in the header.
        cache: Reuses programs already assembled.

    Raises:
        ConfigurationError: Unknown kernel, or the values are outside its domain.
        KeyError: Unknown case.
        ValueError: A multi-stage kernel (several programs; not supported yet).
        RuntimeError: The run faulted or did not halt.
    """
    manifest = default_registry().kernel(kernel)
    if manifest.stages is not None:
        raise ValueError(f"{kernel}: multi-stage kernels are not recorded yet")
    chosen = find_case(manifest, case, values)
    element_format = ElementFormat(fmt) if fmt else chosen.formats[0]
    staged = stage(manifest, chosen, element_format, cache=cache)
    header = {
        "kernel": kernel,
        "case": chosen.name,
        "params": {
            k: v for k, v in staged.plan.params.items() if isinstance(v, (int, float, str, bool))
        },
        "emulator_commit": commit,
    }
    return record_staged(staged, header)
