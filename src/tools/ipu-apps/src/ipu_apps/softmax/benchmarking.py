"""Shared benchmark table helper for the softmax app family.

Each softmax_* app's ``benchmark/benchmark.py`` builds a list of
:class:`BenchRow` (one per swept config) and calls
:func:`print_and_write_table` to render a console table and persist it to
``results.md`` next to the benchmark script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchRow:
    label: str
    cycles: int
    cyc_per_row: float
    mult_utilization: float
    acc_utilization: float
    max_err: float
    correct: bool


def print_and_write_table(app_name: str, rows: list[BenchRow], out_path: Path) -> None:
    header = (
        f"{'config':<28}{'cycles':>10}{'cyc/row':>10}"
        f"{'mult%':>8}{'acc%':>8}{'max_err':>12}{'status':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        status = "PASS" if r.correct else "FAIL"
        lines.append(
            f"{r.label:<28}{r.cycles:>10}{r.cyc_per_row:>10.2f}"
            f"{r.mult_utilization * 100:>7.1f}%{r.acc_utilization * 100:>7.1f}%"
            f"{r.max_err:>12.3e}{status:>8}"
        )
    table = "\n".join(lines)
    print(f"=== {app_name} benchmark ===")
    print(table)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# {app_name} benchmark\n\n```\n{table}\n```\n")
