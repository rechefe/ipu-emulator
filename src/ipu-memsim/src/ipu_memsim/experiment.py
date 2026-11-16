"""Experiments: sweep one or two parameters over a workload. The core API.

An experiment is described by a *spec*, a plain dict (a TOML file on disk)::

    name = "fill_time_at_2mb"

    [workload]
    kind = "network"                       # or "kernel", or "trace"
    network = "src/ipu-memsim/networks/mobilevit_s.toml"
    layer_types = ["mlp", "attention"]     # optional: only these layer types

    [model]
    kind = "cache_unit"

    [fixed]                                # every parameter not swept; the rest
    xmem_mb = 2                            # keep the registry's defaults

    [[sweep]]                              # one or two of these
    param = "fill_cycles_per_bank"
    values = [250, 500, 1000, 2111, 4000, 8000]

    [[sweep]]
    param = "xmem_mb"
    min = 0.5                              # or min / max / steps / scale
    max = 8
    steps = 5
    scale = "log"

    [report]
    threshold_percent = 10                 # the break-even level

:func:`run_sweep` replays the workload once per combination of swept values
and returns a :class:`SweepTable`: one row per workload group (a layer type
of a network, or the kernel) and combination, with the ideal cycles, the
stall cycles and the stall as a percentage of the ideal cycles.
:func:`write_results` saves the table as CSV, the spec that produced it (so
any chart can be made again exactly), the break-even table and the charts.

This module imports no user interface. The command line (``memsim sweep
--spec``) and the browser app (``ipu_memsim.ui``) both call it.

Parameter names, ranges and defaults come from ``ipu_memsim.params``.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ipu_memsim.config import make_model
from ipu_memsim.dram import HBM2_DRAMSIM3_FILL_CYCLES
from ipu_memsim.network import trace_for
from ipu_memsim.params import CONFIRMED, Param, build_config, coerce, param, placeholders_in_use
from ipu_memsim.replay import replay
from ipu_memsim.svgplot import Series, heatmap, line_chart
from ipu_memsim.toml_out import to_toml
from ipu_memsim.trace import Trace, load_trace

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = [
    "SweepTable",
    "coerce",
    "WorkItem",
    "break_even",
    "grid_size",
    "load_spec",
    "load_workload",
    "run_sweep",
    "sweep_values",
    "write_results",
]

#: Ask before running more replays than this (``[report] max_replays``).
DEFAULT_MAX_REPLAYS = 20000


@dataclass
class WorkItem:
    """One trace of the workload.

    Attributes:
        group: The row of the result it adds to (its layer type, or the kernel).
        name: The layer's name.
        count: How many times it occurs.
        trace: Its trace; ``None`` when it could not be recorded.
        overrides: Parameter values this layer fixes (a layer's ``weights``).
        error: Why there is no trace.
    """

    group: str
    name: str
    count: int
    trace: Trace | None
    overrides: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class SweepTable:
    """The result of :func:`run_sweep`.

    Attributes:
        spec: The spec that produced it, with every default filled in.
        params: The swept parameters, one or two.
        rows: One dict per group and combination: ``group``, each swept
            parameter's value, ``ideal_cycles``, ``stall_cycles``,
            ``total_cycles``, ``stall_percent`` and ``note``. A combination
            the model cannot run (too few banks for the tables) has
            ``stall_cycles = None`` and the reason in ``note``.
        placeholders: The ASSUMED and UNKNOWN parameters the numbers rest on.
        seconds: How long the replays took.
        replays: How many replays ran.
    """

    spec: dict[str, Any]
    params: list[Param]
    rows: list[dict[str, Any]]
    placeholders: list[Param]
    seconds: float = 0.0
    replays: int = 0

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(row["group"] for row in self.rows))


def load_spec(path: Path) -> dict[str, Any]:
    """Read a spec from a TOML file."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def sweep_values(kind: str, sweep: Mapping[str, Any]) -> tuple[Param, list[Any]]:
    """The parameter and the values of one ``[[sweep]]`` entry.

    Either ``values = [...]``, or ``min``, ``max``, ``steps`` and optionally
    ``scale = "log"`` (default ``"linear"``). Example: min 250, max 8000,
    steps 6, log gives 250, 500, 1000, 2000, 4000, 8000.

    Raises:
        ValueError: An unknown or unsweepable parameter, or a bad range.
    """
    p = param(kind, sweep["param"])
    if not p.sweepable:
        raise ValueError(f"{p.name} cannot be swept")
    if "values" in sweep:
        raw = list(sweep["values"])
    else:
        low, high, steps = float(sweep["min"]), float(sweep["max"]), int(sweep.get("steps", 5))
        if steps < 2 or high <= low:
            raise ValueError(f"sweep of {p.name}: need max > min and at least 2 steps")
        if sweep.get("scale", "linear") == "log":
            if low <= 0:
                raise ValueError(f"sweep of {p.name}: a log scale needs min > 0")
            raw = [low * (high / low) ** (i / (steps - 1)) for i in range(steps)]
        else:
            raw = [low + (high - low) * i / (steps - 1) for i in range(steps)]
        # Round away float noise (1999.9999 -> 2000); whole numbers for int parameters.
        raw = [round(v) if p.type is int else float(f"{v:.6g}") for v in raw]
        raw = list(dict.fromkeys(raw))
    if not raw:
        raise ValueError(f"sweep of {p.name} has no values")
    return p, [coerce(p, value) for value in raw]


def grid_size(spec: Mapping[str, Any]) -> int:
    """Combinations of swept values in ``spec`` (replays per trace)."""
    kind = spec["model"]["kind"]
    return math.prod(len(sweep_values(kind, s)[1]) for s in spec.get("sweep", []))


def load_workload(spec: Mapping[str, Any], root: Path, commit: str = "unknown") -> list[WorkItem]:
    """The traces a spec's ``[workload]`` names, recorded when missing.

    Recorded traces are kept in ``traces_dir`` (default ``memsim-out/traces``
    under ``root``) and reused: record once, replay many. Only recording
    imports the kernel framework; replaying saved traces needs numpy alone.

    Args:
        spec: The experiment.
        root: The folder relative paths in the spec start from (the checkout).
        commit: The emulator commit, kept in new traces.
    """
    workload = spec["workload"]
    traces = root / spec.get("traces_dir", "memsim-out/traces")
    kind = workload.get("kind", "network")
    if kind == "trace":
        trace = load_trace(root / workload["trace"])
        return [WorkItem(trace.header["kernel"], trace.header["kernel"], 1, trace)]
    if kind == "kernel":
        layers = [{"type": workload["kernel"], "name": workload["kernel"], **workload}]
    elif kind == "network":
        with open(root / workload["network"], "rb") as handle:
            layers = tomllib.load(handle).get("layer", [])
        wanted = workload.get("layer_types")
        if wanted:
            layers = [layer for layer in layers if layer["type"] in wanted]
    else:
        raise ValueError(f"[workload] kind must be network, kernel or trace, not {kind!r}")
    items = []
    for layer in layers:
        count = int(layer.get("count", 1))
        if count == 0:
            continue  # kept in the network file for reference only
        overrides = {"weight_tables": ",".join(layer["weights"])} if "weights" in layer else {}
        item = WorkItem(layer["type"], layer.get("name", layer["kernel"]), count, None, overrides)
        try:
            item.trace = trace_for(layer, traces, commit)
        except Exception as error:  # noqa: BLE001 (any failure becomes the item's note)
            item.error = f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"
        items.append(item)
    return items


def run_sweep(  # noqa: C901 (one loop nest: groups, points, traces)
    spec: Mapping[str, Any],
    root: Path = Path("."),
    *,
    workload: Sequence[WorkItem] | None = None,
    progress: Callable[[int, int], None] | None = None,
    commit: str = "unknown",
) -> SweepTable:
    """Run the experiment ``spec`` describes.

    Args:
        spec: The experiment (see the module docstring).
        root: Where relative paths in the spec start from.
        workload: Traces already loaded (:func:`load_workload`), to reuse
            them over several runs; ``None`` loads them.
        progress: Called as ``progress(done, total)`` after each replay.
        commit: The emulator commit, kept in traces recorded now.

    Returns:
        The table. Its rows are in group order, then the first swept
        parameter's order, then the second's.

    Raises:
        ValueError: The spec names an unknown model or parameter, gives a
            value outside its range, or sweeps more than two parameters.
    """
    kind = spec["model"]["kind"]
    fixed = dict(spec.get("fixed", {}))
    sweeps = [sweep_values(kind, s) for s in spec.get("sweep", [])]
    if not 1 <= len(sweeps) <= 2:
        raise ValueError("an experiment sweeps one or two parameters")
    for p, _ in sweeps:
        if p.name in fixed:
            raise ValueError(f"{p.name} is both swept and fixed")
    build_config(kind, fixed)  # check the fixed values before any work

    items = list(workload) if workload is not None else load_workload(spec, root, commit)
    points: list[dict[str, Any]] = [{}]
    for p, values in sweeps:  # every combination, first parameter outermost
        points = [{**point, p.name: value} for point in points for value in values]

    usable = [item for item in items if item.trace is not None]
    total = len(points) * len(usable)
    done = 0
    started = time.perf_counter()
    rows = []
    groups = list(dict.fromkeys(item.group for item in items))
    for group in groups:
        members = [item for item in usable if item.group == group]
        missing = [item.name for item in items if item.group == group and item.trace is None]
        for point in points:
            ideal = stall = 0
            note = f"not measured: {'; '.join(missing)}" if missing else ""
            failed = ""
            for item in members:
                assert item.trace is not None
                ideal += item.trace.ideal_cycles * item.count
                try:
                    model = make_model(build_config(kind, {**fixed, **item.overrides, **point}))
                    stall += replay(item.trace, model).stall_cycles * item.count
                except ValueError as error:
                    failed = f"{item.name}: {error}"
                done += 1
                if progress is not None:
                    progress(done, total)
            row = {"group": group, **point, "ideal_cycles": ideal}
            if failed or not members:
                row.update(stall_cycles=None, total_cycles=None, stall_percent=None,
                           note=failed or note)  # fmt: skip
            else:
                row.update(stall_cycles=stall, total_cycles=ideal + stall,
                           stall_percent=100.0 * stall / ideal if ideal else 0.0, note=note)  # fmt: skip
            rows.append(row)

    full_spec = {
        "name": spec.get("name", "experiment"),
        "workload": dict(spec["workload"]),
        "model": {"kind": kind},
        "fixed": fixed,
        "sweep": [{"param": p.name, "values": list(values)} for p, values in sweeps],
        "report": {"threshold_percent": 10.0, **spec.get("report", {})},
    }
    if "traces_dir" in spec:
        full_spec["traces_dir"] = spec["traces_dir"]
    return SweepTable(
        spec=full_spec,
        params=[p for p, _ in sweeps],
        rows=rows,
        # The swept parameters are the experiment's subject, not placeholders.
        placeholders=[
            p
            for p in placeholders_in_use(build_config(kind, fixed))
            if p.name not in {swept.name for swept, _ in sweeps}
        ],
        seconds=time.perf_counter() - started,
        replays=done,
    )


def break_even(table: SweepTable, threshold_percent: float | None = None) -> list[dict[str, Any]]:
    """Per group, where the stall first exceeds ``threshold_percent`` of the
    ideal cycles along the *first* swept parameter.

    For each group (and each value of the second parameter, if there is one)
    the result has ``first_above``: the smallest swept value whose stall is
    over the threshold, and ``crossing``: the point between that value and
    the one before it where a straight line between the two crosses the
    threshold (an estimate, not a replay). ``first_above`` is ``None`` when
    no swept value is over the threshold, and ``crossing`` is ``None`` when
    the very first value already is.

    Example: stall is 4% at 1000 cycles and 12% at 2000. The first value
    above 10% is 2000; the line crosses 10% at 1000 + (10 - 4) / (12 - 4) *
    1000 = 1750.
    """
    if threshold_percent is None:
        threshold_percent = float(table.spec["report"]["threshold_percent"])
    if table.params[0].type not in (int, float):
        return []  # a switch or a choice has no "smallest value above"
    first = table.params[0].name
    second = table.params[1].name if len(table.params) > 1 else None
    result = []
    keys = dict.fromkeys((row["group"], row[second] if second else None) for row in table.rows)
    for group, other in keys:
        line = sorted(
            (row for row in table.rows
             if row["group"] == group and (second is None or row[second] == other)
             and row["stall_percent"] is not None),
            key=lambda row: (isinstance(row[first], str), row[first]),  # numbers first, then "auto"
        )  # fmt: skip
        line = [row for row in line if not isinstance(row[first], str)]
        entry: dict[str, Any] = {"group": group}
        if second:
            entry[second] = other
        entry.update(threshold_percent=threshold_percent, first_above=None, crossing=None,
                     max_stall_percent=max((row["stall_percent"] for row in line), default=None))  # fmt: skip
        for index, row in enumerate(line):
            if row["stall_percent"] > threshold_percent:
                entry["first_above"] = row[first]
                if index > 0:
                    before = line[index - 1]
                    share = (threshold_percent - before["stall_percent"]) / (
                        row["stall_percent"] - before["stall_percent"]
                    )
                    entry["crossing"] = before[first] + share * (row[first] - before[first])
                break
        result.append(entry)
    return result


def _is_log(values: Sequence[float]) -> bool:
    """Whether values spread over more than a factor of 20 (drawn on a log axis)."""
    positive = [v for v in values if isinstance(v, (int, float)) and v > 0]
    return len(positive) == len(values) and len(values) > 2 and max(positive) / min(positive) > 20


def _label(p: Param) -> str:
    return f"{p.name} ({p.unit})" if p.unit else p.name


def charts(table: SweepTable) -> dict[str, str]:
    """The table's charts as SVG text, by file name.

    One swept parameter: line charts of stall percent against it, one line
    per group (eight lines per chart at most). Two: one heatmap per group,
    the first parameter across and the second up.
    """
    name = table.spec["name"]
    toy = [p.name for p in table.placeholders if p.status != CONFIRMED]
    subtitle = f"TOY MODEL: {len(toy)} placeholder parameters (listed in spec.toml)"
    if not toy:
        subtitle = "every parameter is CONFIRMED"
    # One labelled reference point on a fill-time axis, not the truth.
    marks = [(float(HBM2_DRAMSIM3_FILL_CYCLES), "DRAMsim3 HBM2 anchor")]
    first = table.params[0]
    out: dict[str, str] = {}
    if len(table.params) == 1:
        numeric = all(isinstance(r[first.name], (int, float)) and not isinstance(r[first.name], bool)
                      for r in table.rows)  # fmt: skip
        if not numeric:
            return out  # a switch or a choice: the table says it all
        lines = []
        for group in table.groups:
            points = [(row[first.name], row["stall_percent"]) for row in table.rows
                      if row["group"] == group and row["stall_percent"] is not None]  # fmt: skip
            if points:
                lines.append(Series(group, points))
        xs = sorted({x for line in lines for x, _ in line.points})
        for start in range(0, len(lines), 8):
            suffix = "" if len(lines) <= 8 else f"_{start // 8 + 1}"
            out[f"{name}{suffix}.svg"] = line_chart(
                lines[start : start + 8],
                title=f"Stall against {first.name}",
                subtitle=subtitle,
                x_label=_label(first),
                y_label="stall (% of ideal cycles)",
                log_x=_is_log(xs),
                marks=marks if first.name == "fill_cycles_per_bank" else (),
            )
        return out
    second = table.params[1]
    xs = list(dict.fromkeys(row[first.name] for row in table.rows))
    ys = list(dict.fromkeys(row[second.name] for row in table.rows))
    for group in table.groups:
        cell = {(row[first.name], row[second.name]): row["stall_percent"]
                for row in table.rows if row["group"] == group}  # fmt: skip
        out[f"{name}_{group}.svg"] = heatmap(
            xs, ys, [[cell.get((x, y)) for x in xs] for y in ys],
            title=f"{group}: stall (% of ideal cycles)",
            subtitle=subtitle, x_label=_label(first), y_label=_label(second),
        )  # fmt: skip
    return out


def write_results(table: SweepTable, out: Path) -> list[Path]:
    """Save a sweep: ``results.csv``, ``spec.toml`` (the exact experiment),
    ``break_even.csv`` and the charts. Returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in table.params]
    columns = [
        "group",
        *names,
        "ideal_cycles",
        "stall_cycles",
        "total_cycles",
        "stall_percent",
        "note",
    ]
    written = [out / "results.csv", out / "spec.toml", out / "break_even.csv"]
    with open(written[0], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in table.rows:
            cells = [row[c] for c in columns]
            percent = row["stall_percent"]
            cells[columns.index("stall_percent")] = "" if percent is None else f"{percent:.3f}"
            writer.writerow(["" if c is None else c for c in cells])
    header = (
        f"# Written by ipu_memsim.experiment: the experiment behind results.csv in this folder.\n"
        f"# Placeholder (ASSUMED or UNKNOWN) parameters in use: "
        f"{', '.join(p.name for p in table.placeholders) or 'none'}.\n"
    )
    written[1].write_text(header + to_toml(table.spec), encoding="utf-8")
    edges = break_even(table)
    with open(written[2], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(edges[0]) if edges else ["group"])
        for edge in edges:
            writer.writerow(["" if v is None else (f"{v:.4g}" if isinstance(v, float) else v)
                             for v in edge.values()])  # fmt: skip
    for filename, svg in charts(table).items():
        (out / filename).write_text(svg, encoding="utf-8")
        written.append(out / filename)
    return written
