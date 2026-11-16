"""Compare the model with measurements from the FPGA, and calibrate it.

**Import format.** One CSV file, one row per measured run::

    kernel,config,measured_cycles,measured_stall_cycles,measured_fetches,measured_bank_fill_cycles,note
    residual_add_default,fpga_16banks,790,465,2,232,

- ``kernel``: the name of a saved trace (its file name without ``.csv.gz``);
- ``config``: the name of a model config (a TOML file's stem) describing how
  the FPGA was set up for that run;
- ``measured_cycles``: cycles from kernel start to kernel done (required);
- the other columns are optional and may be left empty: the stall counter,
  the number of bank fills, and the measured time of one bank fill.

**Metrics** (:func:`compare`), all on total cycles unless a stall counter is
given:

- *relative error* of one run = (model - measured) / measured. +0.05 means
  the model predicts 5% more cycles than were measured.
- *mean absolute relative error* over the runs: the average size of the miss.
- *bias* = the mean signed relative error: positive when the model is
  pessimistic overall.
- *rank correlation* (Spearman): do model and measurement order the runs the
  same way? 1.0 = the same order, 0 = unrelated. It is what matters for
  "which configuration is better", even when the absolute numbers are off.
  It is the ordinary (Pearson) correlation of the two lists of ranks; tied
  values share the average of their ranks.

**Calibration** (:func:`fit_fill_time`). The bank fill time is the model's
main unknown. Total cycles never fall when it grows, so the value that best
explains a set of runs is found by a one-dimensional search: the fill time
minimising the sum of squared relative errors. Fit on some runs, then check
on runs that were *not* used for fitting (:func:`compare` on the rest): a
model that only fits its own calibration data proves nothing.

:func:`synthetic_measurements` makes "measurements" from the model itself
with random noise, to test this code before real data exists. Its output is
labelled SYNTHETIC in every row.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ipu_memsim.config import make_model
from ipu_memsim.replay import replay
from ipu_memsim.trace import Trace

__all__ = [
    "COLUMNS",
    "Measurement",
    "compare",
    "fit_fill_time",
    "load_measurements",
    "predict",
    "report",
    "save_measurements",
    "spearman",
    "synthetic_measurements",
]

COLUMNS = ("kernel", "config", "measured_cycles", "measured_stall_cycles", "measured_fetches",
           "measured_bank_fill_cycles", "note")  # fmt: skip


@dataclass
class Measurement:
    """One measured run (one row of the import file); optional fields are ``None``."""

    kernel: str
    config: str
    cycles: int
    stall_cycles: int | None = None
    fetches: int | None = None
    bank_fill_cycles: float | None = None
    note: str = ""


def load_measurements(path: Path) -> list[Measurement]:
    """Read the import file.

    Raises:
        ValueError: A required column is missing, or a row has no cycle count.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        missing = {"kernel", "config", "measured_cycles"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")
        result = []
        for number, row in enumerate(reader, start=2):
            if not row["measured_cycles"]:
                raise ValueError(f"{path}, line {number}: no measured_cycles")

            def optional(name: str, convert: Any, row: Mapping[str, str] = row) -> Any:
                return convert(row[name]) if row.get(name) else None

            result.append(
                Measurement(
                    row["kernel"], row["config"], int(row["measured_cycles"]),
                    optional("measured_stall_cycles", int), optional("measured_fetches", int),
                    optional("measured_bank_fill_cycles", float), row.get("note") or "",
                )
            )  # fmt: skip
    return result


def save_measurements(measurements: Sequence[Measurement], path: Path, comment: str = "") -> None:
    """Write measurements in the import format (``comment`` as a ``#`` first line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if comment:
            handle.write(f"# {comment}\n")
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for m in measurements:
            writer.writerow(["" if v is None else v for v in
                             (m.kernel, m.config, m.cycles, m.stall_cycles, m.fetches, m.bank_fill_cycles, m.note)])  # fmt: skip


def predict(
    m: Measurement, traces: Mapping[str, Trace], configs: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    """The model's numbers for one measured run: ``cycles``, ``stall_cycles``, ``fetches``.

    Raises:
        KeyError: The run names a trace or a config that was not given.
    """
    result = replay(traces[m.kernel], make_model(configs[m.config]))
    counters = result.counters
    fetches = (
        counters.get("fetches", 0) + counters.get("refetches", 0) + counters.get("prefetches", 0)
    )
    return {"cycles": result.total_cycles, "stall_cycles": result.stall_cycles, "fetches": fetches}


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation of two equally long lists (see the module).
    1.0 for fewer than two values or when either list is constant."""

    def ranks(values: Sequence[float]) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        rank = np.empty(len(values))
        rank[order] = np.arange(1, len(values) + 1)
        for value in set(values):  # tied values share their average rank
            tied = np.asarray(values) == value
            rank[tied] = rank[tied].mean()
        return rank

    if len(first) < 2:
        return 1.0
    a, b = ranks(first), ranks(second)
    if a.std() == 0 or b.std() == 0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def compare(
    measurements: Sequence[Measurement],
    traces: Mapping[str, Trace],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Model against measurement for every run, and the summary metrics.

    Returns:
        ``{"rows": [...], "runs", "mean_abs_rel_error", "bias",
        "max_abs_rel_error", "rank_correlation"}``. Each row has the measured
        and the model's cycles, ``rel_error``, and the same for stall cycles
        and fetch counts where they were measured.
    """
    rows = []
    for m in measurements:
        model = predict(m, traces, configs)
        row: dict[str, Any] = {
            "kernel": m.kernel, "config": m.config, "measured_cycles": m.cycles, "model_cycles": model["cycles"],
            "rel_error": (model["cycles"] - m.cycles) / m.cycles,
            "measured_stall_cycles": m.stall_cycles, "model_stall_cycles": model["stall_cycles"],
            "measured_fetches": m.fetches, "model_fetches": model["fetches"],
            "fetch_count_matches": None if m.fetches is None else m.fetches == model["fetches"],
        }  # fmt: skip
        rows.append(row)
    errors = [row["rel_error"] for row in rows]
    return {
        "rows": rows,
        "runs": len(rows),
        "mean_abs_rel_error": float(np.mean(np.abs(errors))) if errors else 0.0,
        "bias": float(np.mean(errors)) if errors else 0.0,
        "max_abs_rel_error": float(np.max(np.abs(errors))) if errors else 0.0,
        "rank_correlation": spearman(
            [r["measured_cycles"] for r in rows], [r["model_cycles"] for r in rows]
        ),
    }


def _with_fill(config: Mapping[str, Any], fill_cycles: float) -> dict[str, Any]:
    """``config`` with its bank fill time replaced."""
    return {**config, "dram": {"kind": "bank_fill", "fill_cycles_per_bank": fill_cycles}}


def fit_fill_time(
    measurements: Sequence[Measurement],
    traces: Mapping[str, Trace],
    configs: Mapping[str, Mapping[str, Any]],
    low: float = 0.0,
    high: float = 100_000.0,
    steps: int = 40,
) -> tuple[float, float]:
    """The bank fill time (cycles per 1024-row bank) that best explains the runs.

    The cost of a fill time is the sum of squared relative errors of the
    model's total cycles. It is searched by repeatedly shrinking the interval
    ``low..high`` around the best of three trial points (a ternary search),
    which works because the cost has one minimum when cycles never fall as
    the fill time grows.

    Returns:
        ``(fill_cycles_per_bank, cost)``.
    """

    def cost(fill: float) -> float:
        total = 0.0
        for m in measurements:
            model = replay(
                traces[m.kernel], make_model(_with_fill(configs[m.config], fill))
            ).total_cycles
            total += ((model - m.cycles) / m.cycles) ** 2
        return total

    for _ in range(steps):
        one_third, two_thirds = low + (high - low) / 3, high - (high - low) / 3
        if cost(one_third) <= cost(two_thirds):
            high = two_thirds
        else:
            low = one_third
    best = (low + high) / 2
    return best, cost(best)


def synthetic_measurements(
    runs: Sequence[tuple[str, str]],
    traces: Mapping[str, Trace],
    configs: Mapping[str, Mapping[str, Any]],
    fill_cycles: float,
    noise: float = 0.02,
    seed: int = 0,
) -> list[Measurement]:
    """SYNTHETIC "measurements": the model's own answer at ``fill_cycles`` per
    bank, with each cycle count multiplied by a random factor within
    ``1 +- noise``. For testing the comparison code only."""
    randomness = random.Random(seed)
    result = []
    for kernel, config in runs:
        try:
            run = replay(traces[kernel], make_model(_with_fill(configs[config], fill_cycles)))
        except ValueError:
            continue  # this configuration cannot run the kernel (too few banks for its tables)
        factor = 1.0 + randomness.uniform(-noise, noise)
        cycles = max(1, round(run.total_cycles * factor))
        fetches = run.counters["fetches"] + run.counters["refetches"] + run.counters["prefetches"]
        result.append(
            Measurement(kernel, config, cycles, max(0, cycles - run.ideal_cycles), fetches, fill_cycles,
                        f"SYNTHETIC: model at {fill_cycles:g} cycles per bank, noise {noise:g}, seed {seed}")
        )  # fmt: skip
    return result


def report(comparison: Mapping[str, Any], title: str, fitted: float | None = None) -> str:
    """The comparison as a Markdown report, with the metric definitions."""
    lines = [
        f"# {title}",
        "",
        f"Runs: {comparison['runs']}. Mean absolute relative error: {100 * comparison['mean_abs_rel_error']:.2f}%. "
        f"Bias: {100 * comparison['bias']:+.2f}%. Largest: {100 * comparison['max_abs_rel_error']:.2f}%. "
        f"Rank correlation: {comparison['rank_correlation']:.3f}.",
    ]
    if fitted is not None:
        lines.append(
            f"Bank fill time fitted on the calibration runs: {fitted:.0f} cycles per 1024-row bank."
        )
    lines += [
        "",
        "Definitions. Relative error = (model - measured) / measured, on cycles from kernel start to kernel",
        "done. Bias = the mean of the signed errors (positive: the model predicts more cycles). Rank",
        "correlation (Spearman) = the correlation of the two orderings of the runs: 1.0 means the model",
        "orders the runs exactly as the measurements do.",
        "",
        "Largest disagreements first:",
        "",
        "| kernel | config | measured cycles | model cycles | relative error | measured stall | model stall | fetch count matches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(comparison["rows"], key=lambda r: -abs(r["rel_error"])):
        stall = "" if row["measured_stall_cycles"] is None else row["measured_stall_cycles"]
        match = (
            ""
            if row["fetch_count_matches"] is None
            else ("yes" if row["fetch_count_matches"] else "NO")
        )
        lines.append(
            f"| {row['kernel']} | {row['config']} | {row['measured_cycles']} | {row['model_cycles']} | "
            f"{100 * row['rel_error']:+.2f}% | {stall} | {row['model_stall_cycles']} | {match} |"
        )
    return "\n".join(lines) + "\n"
