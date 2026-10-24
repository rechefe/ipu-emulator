"""Benchmarks: a kernel timed at the sizes its ``[benchmark]`` lists (E06-11).

Each of ``[benchmark] points`` is a set of parameter values over the default
case's, run once per format through :func:`ipu_harness.runner.run_kernel`,
with an alias profile, and held to its golden like any case, so a benchmark
cannot report a fast wrong answer (so a ``[benchmark]`` needs a reference or
op: a point's shapes are its own, so no expected file fits it). ``per`` adds a
cycles-per column and ``max_cycles`` sets the limit (the model's by default).
A point that is wrong for one of the formats asked for stops the run with its
error.

The report's columns: cycles, ``cyc/<per>``, the MULT and ACC
slots' busy cycles, the multiplier lanes that carried data, the part of those
that multiplied by a constant 1 (data routing, not MACs) and the peak MACs
retired, from the model's statistics (reference-model API 1.1.0); then the ISA
aliases observed in each run. ``python -m ipu_harness.cli KERNEL --benchmark``
prints it (``bazel run //kernels/<family>/<kernel> -- --benchmark``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, at_values, expand
from ipu_harness.manifest import BenchmarkDecl, KernelManifest
from ipu_harness.runner import RunOptions, RunResult, run_kernel
from ipu_harness.variants import Variant

__all__ = ["LEGEND", "BenchRow", "point_cases", "run_benchmark", "to_json", "to_markdown"]

#: What the report's columns mean.
LEGEND = (
    "mult% / acc%: cycles the slot was busy. lanes%: multiplier lanes that "
    "carried data. ident%: the part of those lanes that multiplied by a constant 1 "
    "(data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: "
    "verified occurrences of each alias in docs/content/explanation/isa-idioms.md, "
    "subtypes summed (+N? = unverified candidates); occurrences can overlap, so "
    "they are not savings estimates."
)


@dataclass(frozen=True)
class BenchRow:
    """One point, run in one format.

    Attributes:
        label: The point's values (``rows=16,width=128``).
        result: The run.
        per_unit: Cycles over the point's ``per`` parameter, if the benchmark
            names one.
        aliases: Each observed alias: (verified, candidate) occurrences.
        values: The point's values.
    """

    label: str
    result: RunResult
    per_unit: float | None
    aliases: Mapping[str, tuple[int, int]]
    values: Mapping[str, object] = MappingProxyType({})

    def utilization(self) -> dict[str, float] | None:
        """The busy and lane fractions (mult, acc, lanes, ident, effMAC).

        Returns:
            Each as 0..1; ``None`` without the model's statistics (a
            multi-stage kernel).
        """
        stats = self.result.stats
        if stats is None:
            return None
        cycles = stats.cycles or 1
        lanes = stats.mult_lane_ops or 1
        return {
            "mult": stats.mult_active_cycles / cycles,
            "acc": stats.acc_active_cycles / cycles,
            "lanes": stats.lane_occupancy,
            "ident": stats.mult_identity_lane_ops / lanes,
            "effMAC": stats.effective_mac_utilization,
        }


def _label(values: Mapping[str, object]) -> str:
    """A point's values, as its case name and report row show them."""
    return ",".join(f"{k}={v}" for k, v in values.items())


def _decl(manifest: KernelManifest) -> BenchmarkDecl:
    if manifest.benchmark is None:
        raise ConfigurationError(f"{manifest.name} has no [benchmark]")
    return manifest.benchmark


def point_cases(manifest: KernelManifest) -> tuple[Case, ...]:
    """The benchmark's runs, as cases: ``benchmark[<values>]``, each point
    over the default case (its first point, if it sweeps), with its
    generator but not its files (a point's shapes differ).

    Args:
        manifest: A kernel with a ``[benchmark]``.

    Returns:
        A case per point, in order.

    Raises:
        ConfigurationError: The kernel has no ``[benchmark]``.
    """
    bench = _decl(manifest)
    default = next(c for c in expand(manifest) if c.decl.name == "default")
    return tuple(at_values(default, f"benchmark[{_label(p)}]", p) for p in bench.points)


def _observed_aliases(profile: Any) -> dict[str, tuple[int, int]]:
    observed = {}
    for alias, entry in profile.to_dict()["aliases"].items():
        counts = {"verified": 0, "candidate": 0}
        for row in entry["measurements"]:
            counts[row["status"]] += row["count"]
        if any(counts.values()):
            observed[alias] = (counts["verified"], counts["candidate"])
    return observed


def run_benchmark(
    manifest: KernelManifest, fmt: Variant, *, cache: AssemblyCache | None = None
) -> tuple[BenchRow, ...]:
    """Every point of the kernel's ``[benchmark]``, run in ``fmt``.

    Args:
        manifest: A kernel with a ``[benchmark]``.
        fmt: The element format to run in.
        cache: Reuses programs already assembled.

    Returns:
        A row per point. A row whose ``result`` did not pass is a wrong answer
        or a fault: the report shows it and the command line fails on it.

    Raises:
        ConfigurationError: The kernel has no ``[benchmark]``, or a point is
            wrong for it (as ``run_kernel``).
    """
    bench = _decl(manifest)
    options = RunOptions(max_cycles=bench.max_cycles, alias_profile=True)
    cache = cache or AssemblyCache()
    rows = []
    for point, case in zip(bench.points, point_cases(manifest), strict=True):
        result = run_kernel(manifest, case, fmt, options, cache=cache)
        per_unit = None
        if bench.per is not None:
            count = case.values[bench.per]
            assert isinstance(count, int)  # the loader checked per is an int parameter
            per_unit = result.cycles / count
        aliases = _observed_aliases(result.alias_profile) if result.alias_profile else {}
        rows.append(
            BenchRow(
                _label(point),
                result,
                per_unit,
                MappingProxyType(aliases),
                MappingProxyType(dict(point)),
            )
        )
    return tuple(rows)


def _alias_order(alias: str) -> tuple[float, str]:
    number = re.match(r"A(\d+)", alias)
    return (int(number[1]) if number else float("inf"), alias)


def _markdown(headers: Sequence[str], cells: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def to_markdown(rows: Sequence[BenchRow], per: str | None = None) -> str:
    """The report: the metrics table, the observed aliases per point, the legend.

    Args:
        rows: From :func:`run_benchmark`, any formats.
        per: The benchmark's ``per`` parameter, for a ``cyc/<per>`` column.

    Returns:
        Markdown.
    """
    metrics = ("mult", "acc", "lanes", "ident", "effMAC")
    headers = [
        "point",
        "format",
        "result",
        "cycles",
        *([f"cyc/{per}"] if per else []),
        *(f"{m}%" for m in metrics),
    ]
    cells = []
    for row in rows:
        used = row.utilization()
        per_cell = f"{row.per_unit:.2f}" if row.per_unit is not None else "-"
        cells.append(
            [
                row.label,
                row.result.format.value,
                "pass" if row.result.passed else "FAIL",
                str(row.result.cycles),
                *([per_cell] if per else []),
                *(f"{used[m] * 100:.1f}%" if used else "-" for m in metrics),
            ]
        )
    text = _markdown(headers, cells)
    aliases = sorted({a for row in rows for a in row.aliases}, key=_alias_order)
    if aliases:
        alias_cells = [
            [f"{row.label} ({row.result.format.value})"]
            + [
                (f"{v}" + (f" +{c}?" if c else "")) if (v, c) != (0, 0) else ""
                for v, c in (row.aliases.get(a, (0, 0)) for a in aliases)
            ]
            for row in rows
        ]
        text += "\n\nISA aliases\n\n" + _markdown(["point", *aliases], alias_cells)
    else:
        text += "\n\nISA aliases: none observed."
    return text + "\n\n" + LEGEND + "\n"


def _slug(text: str) -> str:
    """``text`` as a benchmark-name part: lower case, ``[a-z0-9_.]`` only."""
    return re.sub(r"[^a-z0-9_.]+", "_", text.lower().replace("-", "m")).strip("_")


def to_json(rows: Sequence[BenchRow]) -> str:
    """The rows in the nightly history's format (E01-14, ``ipu_bench``'s
    ``Result``, github-action-benchmark's ``customSmallerIsBetter``): one
    entry per row, ``kernel/<kernel>/<format>/<point>`` in cycles, the rest
    in ``extra``.

    Args:
        rows: From :func:`run_benchmark`.

    Returns:
        A JSON list.
    """
    entries = []
    for row in rows:
        result = row.result
        extra = {
            "kernel": result.kernel,
            "format": result.format.value,
            "values": {k: v for k, v in row.values.items()},
            "passed": result.passed,
            "per_unit": row.per_unit,
            "utilization": row.utilization(),
            "aliases": {a: list(n) for a, n in row.aliases.items()},
        }
        point = "_".join(f"{_slug(str(k))}_{_slug(str(v))}" for k, v in row.values.items())
        entries.append(
            {
                "name": f"kernel/{_slug(result.kernel)}/{result.format.value}/{point or 'default'}",
                "unit": "cycles",
                "value": result.cycles,
                "extra": json.dumps(extra, sort_keys=True),
            }
        )
    return json.dumps(entries, indent=2)
