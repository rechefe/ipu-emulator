"""Benchmarks: a kernel timed at the sizes its ``[benchmark]`` lists (E06-11).

``[benchmark] points`` replaces a legacy ``benchmark.py``'s ``CONFIGS``: each
point is a set of parameter values over the default case's, run once per
format through :func:`ipu_harness.runner.run_kernel`, with an alias profile,
and held to its golden like any case, so a benchmark cannot report a fast
wrong answer. ``per`` (``PER``) adds a cycles-per column and ``max_cycles``
(``MAX_CYCLES``) raises the limit.

The report has the legacy columns: cycles, ``cyc/<per>``, the MULT and ACC
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
from ipu_harness.cases import Case, expand
from ipu_harness.manifest import CaseDecl, KernelManifest
from ipu_harness.runner import RunOptions, RunResult, run_kernel
from ipu_isa.formats import ElementFormat

__all__ = ["LEGEND", "BenchRow", "points", "render", "run_benchmark", "to_json"]

#: What the report's columns mean.
LEGEND = (
    "mult% / acc%: cycles the slot was busy. lanes%: multiplier lanes that "
    "carried data. ident%: the part of those lanes that multiplied by a constant 1 "
    "(data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: "
    "verified occurrences of each alias in docs/content/isa-alias-catalogue.md, "
    "subtypes summed (+N? = unverified candidates); occurrences can overlap, so "
    "they are not savings estimates."
)


@dataclass(frozen=True)
class BenchRow:
    """One point, run in one format.

    Attributes:
        label: The point's values (``rows=16, width=128``).
        result: The run.
        per_unit: Cycles over the point's ``per`` parameter, if the benchmark
            names one.
        aliases: Each observed alias: (verified, candidate) occurrences.
    """

    label: str
    result: RunResult
    per_unit: float | None
    aliases: Mapping[str, tuple[int, int]]

    def utilization(self) -> dict[str, float] | None:
        """mult, acc, lanes, ident and effMAC as fractions; ``None`` without
        the model's statistics (a multi-stage kernel)."""
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
    return ", ".join(f"{k}={v}" for k, v in values.items())


def points(manifest: KernelManifest) -> tuple[Case, ...]:
    """The benchmark's runs, as cases: each point over the default case's
    values, with its generator (a point's shapes differ, so not its files).

    Raises:
        ConfigurationError: The kernel has no ``[benchmark]``.
    """
    if manifest.benchmark is None:
        raise ConfigurationError(f"{manifest.name} has no [benchmark]")
    default = next(c for c in expand(manifest) if c.decl.name == "default")
    cases = []
    for point in manifest.benchmark.points:
        name = f"benchmark[{','.join(f'{k}={v}' for k, v in point.items())}]"
        values = MappingProxyType({**default.values, **point})
        cases.append(
            Case(
                name=name,
                values=values,
                formats=default.formats,
                inputs=MappingProxyType({}),
                expected=MappingProxyType({}),
                generator=default.generator,
                tier="default",
                decl=CaseDecl(name, values, generator=default.generator),
            )
        )
    return tuple(cases)


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
    manifest: KernelManifest, fmt: ElementFormat, *, cache: AssemblyCache | None = None
) -> tuple[BenchRow, ...]:
    """Every point of the kernel's ``[benchmark]``, run in ``fmt``.

    Returns:
        A row per point; a row whose ``result`` did not pass is a wrong answer
        (or a fault), which the report shows and the command line fails on.

    Raises:
        ConfigurationError: The kernel has no ``[benchmark]``, or a point is
            wrong for it.
    """
    cases = points(manifest)  # raises without a [benchmark]
    bench = manifest.benchmark
    assert bench is not None
    options = RunOptions(max_cycles=bench.max_cycles, alias_profile=True)
    cache = cache or AssemblyCache()
    rows = []
    for point, case in zip(bench.points, cases, strict=True):
        result = run_kernel(manifest, case, fmt, options, cache=cache)
        per_unit = None
        if bench.per is not None:
            count = case.values[bench.per]
            assert isinstance(count, int)  # the loader checked per is an int parameter
            per_unit = result.cycles / count
        aliases = _observed_aliases(result.alias_profile) if result.alias_profile else {}
        rows.append(BenchRow(_label(point), result, per_unit, MappingProxyType(aliases)))
    return tuple(rows)


def _alias_order(alias: str) -> tuple[float, str]:
    number = re.match(r"A(\d+)", alias)
    return (int(number[1]) if number else float("inf"), alias)


def _markdown(headers: Sequence[str], cells: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in cells]
    return "\n".join(lines)


def render(rows: Sequence[BenchRow], per: str | None = None) -> str:
    """The metrics table, then the observed aliases per point, in Markdown."""
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
        cells.append(
            [
                row.label,
                row.result.format.value,
                "pass" if row.result.passed else "FAIL",
                str(row.result.cycles),
                *([f"{row.per_unit:.2f}"] if per and row.per_unit is not None else []),
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
    return text + "\n\n" + LEGEND + "\n"


def to_json(rows: Sequence[BenchRow]) -> str:
    """The rows as JSON: one object per point, for the nightly history (E01-14)."""
    return json.dumps(
        [
            {
                "point": row.label,
                "format": row.result.format.value,
                "passed": row.result.passed,
                "cycles": row.result.cycles,
                "per_unit": row.per_unit,
                "utilization": row.utilization(),
                "aliases": {a: list(n) for a, n in row.aliases.items()},
            }
            for row in rows
        ],
        indent=2,
    )
