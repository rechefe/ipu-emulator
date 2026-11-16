"""Sensitivity: which assumptions actually move the results?

Many results rest on ASSUMED parameters. This module varies them **one at a
time**: each parameter takes a low and a high setting while every other
parameter stays at the baseline, and the stall of every workload group (layer
type) is recorded. The *spread* of a parameter for a group is the largest
minus the smallest stall percentage over its settings; a parameter whose
spread is near zero cannot change the conclusions, whatever its true value.

A sensitivity spec is an experiment spec (``ipu_memsim.experiment``) whose
``[[sweep]]`` entries are replaced by ``[[vary]]`` entries::

    [fixed]                       # the baseline every row starts from
    xmem_mb = 2
    fill_ns_per_bank = 2111

    [[vary]]
    param = "prefetch_distance"
    low = 0
    high = 16
    reason = "0: no prefetch at all. 16: the spec's DMA fills every free bank."

The baseline value of a parameter is what ``[fixed]`` gives, else the
registry default. "low" and "high" are just two other settings: for a switch
they are its alternatives, in no particular order.

Example of reading the result: if ``prefetch_distance`` gives 4.3% / 4.3% /
4.3% for attention, its spread is 0 and the question "how far does the DMA
prefetch" does not matter for attention at this operating point. If
``residency`` gives 0% / 102% for elementwise, its spread is 102 and that
question comes first.

Each parameter is one call of ``experiment.run_sweep`` with the three
settings as its swept values, so the numbers are the same ones a sweep gives.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipu_memsim import experiment
from ipu_memsim.params import param
from ipu_memsim.toml_out import to_toml

__all__ = ["SensitivityRow", "rank", "run_pair", "run_sensitivity", "write_sensitivity"]


@dataclass
class SensitivityRow:
    """One parameter's effect on one group.

    Attributes:
        param: The parameter varied.
        group: The workload group (layer type).
        settings: The low, baseline and high settings.
        stall_percent: The stall at each, in percent of ideal cycles; ``None``
            where the configuration cannot run the group.
        spread: Largest minus smallest of the stalls that exist.
        status: The parameter's registry label.
        reason: Why this range (from the spec).
    """

    param: str
    group: str
    settings: tuple[Any, Any, Any]
    stall_percent: tuple[float | None, float | None, float | None]
    spread: float
    status: str
    reason: str


def run_sensitivity(
    spec: Mapping[str, Any],
    root: Path = Path("."),
    *,
    workload: Sequence[experiment.WorkItem] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SensitivityRow]:
    """Vary each ``[[vary]]`` parameter of ``spec`` alone; one row per
    parameter and group, in the spec's order.

    Raises:
        ValueError: A parameter is unknown, or its low or high setting is
            outside its range.
    """
    kind = spec["model"]["kind"]
    fixed = dict(spec.get("fixed", {}))
    items = list(workload) if workload is not None else experiment.load_workload(spec, root)
    rows = []
    for vary in spec.get("vary", []):
        p = param(kind, vary["param"])
        baseline = fixed.get(p.name, p.default)
        settings = (vary["low"], baseline, vary["high"])
        if progress is not None:
            progress(p.name)
        # The three settings as one sweep; the parameter leaves [fixed] for it.
        one = {
            "name": spec.get("name", "sensitivity"),
            "workload": spec["workload"],
            "model": {"kind": kind},
            "fixed": {k: v for k, v in fixed.items() if k != p.name},
            "sweep": [{"param": p.name, "values": list(dict.fromkeys(settings))}],
        }
        table = experiment.run_sweep(one, root, workload=items)
        for group in table.groups:
            by_setting = {
                row[p.name]: row["stall_percent"] for row in table.rows if row["group"] == group
            }
            # run_sweep stores values as the registry types them (16 -> 16).
            stalls = tuple(by_setting.get(experiment.coerce(p, s)) for s in settings)
            known = [s for s in stalls if s is not None]
            rows.append(
                SensitivityRow(
                    p.name, group, settings, stalls,  # type: ignore[arg-type]
                    max(known) - min(known) if known else 0.0, p.status, vary.get("reason", ""),
                )
            )  # fmt: skip
    return rows


def rank(rows: Sequence[SensitivityRow]) -> list[dict[str, Any]]:
    """The parameters ordered by their largest spread over the groups.

    Each entry: ``param``, ``status``, ``max_spread`` and the group it occurs
    in, ``groups_moved`` (how many groups move by more than 1 percentage
    point), and ``reason``.
    """
    ranking = []
    for name in dict.fromkeys(row.param for row in rows):
        mine = [row for row in rows if row.param == name]
        top = max(mine, key=lambda row: row.spread)
        ranking.append(
            {
                "param": name,
                "status": top.status,
                "max_spread": top.spread,
                "in_group": top.group,
                "groups_moved": sum(1 for row in mine if row.spread > 1.0),
                "low": top.settings[0],
                "default": top.settings[1],
                "high": top.settings[2],
                "reason": top.reason,
            }
        )
    return sorted(ranking, key=lambda entry: -entry["max_spread"])


def run_pair(
    spec: Mapping[str, Any],
    first: str,
    second: str,
    root: Path = Path("."),
    *,
    workload: Sequence[experiment.WorkItem] | None = None,
) -> experiment.SweepTable:
    """Vary two of the spec's ``[[vary]]`` parameters together (low, baseline
    and high of each: a 3 x 3 grid), as an ordinary two-parameter sweep."""
    kind = spec["model"]["kind"]
    fixed = dict(spec.get("fixed", {}))
    sweeps = []
    for name in (first, second):
        vary = next(v for v in spec["vary"] if v["param"] == name)
        p = param(kind, name)
        values = [vary["low"], fixed.get(name, p.default), vary["high"]]
        sweeps.append({"param": name, "values": list(dict.fromkeys(values))})
    pair = {
        "name": f"{spec.get('name', 'sensitivity')}_pair",
        "workload": spec["workload"],
        "model": {"kind": kind},
        "fixed": {k: v for k, v in fixed.items() if k not in (first, second)},
        "sweep": sweeps,
    }
    return experiment.run_sweep(pair, root, workload=workload)


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def write_sensitivity(
    spec: Mapping[str, Any],
    rows: Sequence[SensitivityRow],
    out: Path,
    pair: experiment.SweepTable | None = None,
) -> list[Path]:
    """Save a sensitivity run: ``sensitivity.csv`` (every parameter and
    group), ``ranking.csv`` and ``ranking.md`` (parameters by largest spread),
    ``spec.toml`` (the exact run) and, with ``pair``, its heatmaps in
    ``pair/``. Returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "sensitivity.csv", out / "ranking.csv", out / "ranking.md", out / "spec.toml"]
    with open(written[0], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["param", "status", "group", "low", "default", "high",
                         "stall_low", "stall_default", "stall_high", "spread"])  # fmt: skip
        for row in rows:
            writer.writerow([row.param, row.status, row.group, *row.settings,
                             *(_cell(s) for s in row.stall_percent), f"{row.spread:.2f}"])  # fmt: skip
    ranking = rank(rows)
    with open(written[1], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(ranking[0]) if ranking else ["param"])
        for entry in ranking:
            writer.writerow([f"{v:.2f}" if isinstance(v, float) else v for v in entry.values()])
    lines = [
        f"# Sensitivity ranking: {spec.get('name', 'sensitivity')}",
        "",
        "One parameter varied at a time, the others at the baseline. Spread = largest minus",
        "smallest stall, in percentage points of ideal cycles, for the layer type it moves most.",
        "TOY MODEL results: they rank the questions, they are not chip numbers.",
        "",
        "| rank | parameter | status | settings (low / default / high) | largest spread | in layer type | layer types moved > 1 pt |",
        "|---|---|---|---|---|---|---|",
    ]
    for position, entry in enumerate(ranking, start=1):
        lines.append(
            f"| {position} | `{entry['param']}` | {entry['status']} | {entry['low']} / {entry['default']} / "
            f"{entry['high']} | {entry['max_spread']:.2f} | {entry['in_group']} | {entry['groups_moved']} |"
        )
    written[2].write_text("\n".join(lines) + "\n", encoding="utf-8")
    written[3].write_text(to_toml(dict(spec)), encoding="utf-8")
    if pair is not None:
        written += experiment.write_results(pair, out / "pair")
    return written
