"""Benchmark a kernel by running its own cases at several sizes.

A kernel package's optional ``benchmark.py`` is a declaration, like
``cases.py``. ``CONFIGS`` lists overrides of the default case's options.
Optionally, ``PER`` names the option cycles are divided by (``PER = "rows"``
adds a ``cyc/rows`` column) and ``MAX_CYCLES`` raises the case's cycle limit.

``bazel run :benchmark_<package>`` runs every config through :func:`run_case`
-- assembled, constructed through the registry and checked exactly as the tests
do, so a benchmark cannot report a fast wrong answer -- prints the tables, and
writes them to ``results.md`` beside ``benchmark.py``.

Besides cycles, the tables carry the run's MAC accounting (``RunStats``) and
the full ISA alias profile (``AliasProfile``, every alias in
``docs/content/isa-alias-catalogue.md``): raw stage occupancy overstates work,
because kernels route data through the multiplier. Profiling never changes
cycles or stats.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

from ipu_emu.alias_profile import AliasProfile

from ipu_apps.kernel_registry.cases import load_cases, options_label, package_kernel, run_case

LEGEND = (
    "mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that "
    "carried data. ident%: busy multiply cycles that were identity multiplies "
    "(data routing, not MACs). effMAC%: peak MACs actually retired. "
    "ISA aliases: verified occurrences of each alias in "
    "docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified "
    "candidates); occurrences can overlap, so they are not savings estimates."
)


@dataclass(frozen=True)
class BenchRow:
    label: str
    cycles: int
    per_unit: float | None
    mult_utilization: float
    acc_utilization: float
    lane_occupancy: float
    identity_fraction: float
    effective_mac_utilization: float
    # alias id -> (verified, candidate) occurrences, observed aliases only.
    aliases: Mapping[str, tuple[int, int]]


def run_benchmark(module) -> list[BenchRow]:
    per = getattr(module, "PER", None)
    kernel = package_kernel(module.__name__.rpartition(".")[0])
    case = load_cases(kernel)["default"]
    rows = []
    for options in module.CONFIGS:
        profile = AliasProfile()
        state, cycles = run_case(kernel, case, options=options, alias_profile=profile,
                                 max_cycles=getattr(module, "MAX_CYCLES", None))
        stats = state.stats
        rows.append(BenchRow(
            label=options_label(options),
            cycles=cycles,
            per_unit=cycles / {**case.defaults, **options}[per] if per else None,
            mult_utilization=stats.mult_utilization,
            acc_utilization=stats.acc_utilization,
            lane_occupancy=stats.lane_occupancy,
            identity_fraction=stats.identity_fraction,
            effective_mac_utilization=stats.effective_mac_utilization,
            aliases=_observed_aliases(profile),
        ))
        # Release this run's state (and its XMEM) before the next one is built.
        del state
    return rows


def _observed_aliases(profile: AliasProfile) -> dict[str, tuple[int, int]]:
    observed = {}
    for alias, entry in profile.to_dict()["aliases"].items():
        counts = {"verified": 0, "candidate": 0}
        for row in entry["measurements"]:
            counts[row["status"]] += row["count"]
        if any(counts.values()):
            observed[alias] = (counts["verified"], counts["candidate"])
    return observed


def _table(labels: Sequence[str], headers: Sequence[str], cells: Sequence[Sequence[str]]) -> str:
    first = max([26] + [len(label) for label in labels]) + 2
    widths = [max([len(h)] + [len(row[i]) for row in cells]) + 2 for i, h in enumerate(headers)]
    lines = ["config".ljust(first) + "".join(h.rjust(w) for h, w in zip(headers, widths))]
    lines.append("-" * len(lines[0]))
    for label, row in zip(labels, cells):
        lines.append(label.ljust(first) + "".join(c.rjust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def _alias_order(alias: str):
    number = re.match(r"A(\d+)", alias)
    return (int(number[1]) if number else float("inf"), alias)


def render_table(rows: Sequence[BenchRow], per: str | None = None) -> str:
    """The metrics table, then a table of observed ISA aliases per config."""
    labels = [r.label for r in rows]
    headers = ["cycles"] + ([f"cyc/{per}"] if per else []) + [
        "mult%", "acc%", "lanes%", "ident%", "effMAC%"]
    cells = [[str(r.cycles)] + ([f"{r.per_unit:.2f}"] if per else []) + [
        f"{value * 100:.1f}%" for value in (
            r.mult_utilization, r.acc_utilization, r.lane_occupancy,
            r.identity_fraction, r.effective_mac_utilization)]
        for r in rows]
    aliases = sorted({alias for r in rows for alias in r.aliases}, key=_alias_order)
    if not aliases:
        return _table(labels, headers, cells) + "\n\nISA aliases: none observed"
    hits = [[_occurrences(*r.aliases.get(alias, (0, 0))) for alias in aliases] for r in rows]
    names = "  ".join(alias.replace("_", " ", 1) for alias in aliases)
    return (_table(labels, headers, cells) + f"\n\nISA aliases: {names}\n"
            + _table(labels, [alias.partition("_")[0] for alias in aliases], hits))


def _occurrences(verified: int, candidate: int) -> str:
    return f"{verified}+{candidate}?" if candidate else str(verified)


def main(argv=None) -> int:
    """``argv``: the benchmark module's dotted name and its workspace path."""
    module_name, package_dir = sys.argv[1:] if argv is None else argv
    module = import_module(module_name)
    title = f"{package_dir.rpartition('/')[2]} benchmark"
    table = render_table(run_benchmark(module), getattr(module, "PER", None))
    print(f"=== {title} ===")
    print(table)
    # `bazel run` marks the source tree; results.md is committed beside benchmark.py.
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        results = Path(workspace) / package_dir / "results.md"
        results.write_text(f"# {title}\n\n```\n{table}\n```\n\n{LEGEND}\n")
        print(f"wrote {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
