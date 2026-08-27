"""Shared benchmark reporting helpers for convolutions_universal apps.

Each app's ``benchmark/benchmark.py`` runs a fixed CONFIGS list through the
emulator, collects one :class:`BenchRow` per config (real cycle count and
real MULT-slot utilization, read from ``state.stats`` -- not an analytical
estimate), and hands the rows to :func:`print_and_write_table` for a console
table plus a persisted markdown file.

Usage in a per-app benchmark.py::

    from ipu_apps.convolutions_universal.benchmarking import (
        BenchRow, print_and_write_table,
    )

    rows = []
    for cfg in CONFIGS:
        state, cycles, mismatches = run_config(...)
        rows.append(BenchRow(
            label="16x16x8", cycles=cycles,
            mult_utilization=state.stats.mult_utilization,
            correct=(mismatches == 0),
        ))
    print_and_write_table("depthwise_conv_universal", rows, out_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchRow:
    """One benchmark config's result."""

    label: str
    cycles: int
    mult_utilization: float  # 0.0-1.0, from state.stats.mult_utilization
    correct: bool
    elapsed_s: float = 0.0


def format_console_table(app_name: str, rows: list[BenchRow]) -> str:
    header = (
        f"{'config':>22} {'cycles':>10} {'mult%':>7} "
        f"{'corr':>5} {'time(s)':>8}"
    )
    lines = [f"=== {app_name} ===", header, "-" * len(header)]
    total_cycles = 0
    total_weighted_util = 0.0
    all_correct = True
    for row in rows:
        lines.append(
            f"{row.label:>22} {row.cycles:>10} "
            f"{row.mult_utilization * 100:>6.1f}% "
            f"{'PASS' if row.correct else 'FAIL':>5} {row.elapsed_s:>8.2f}"
        )
        total_cycles += row.cycles
        total_weighted_util += row.mult_utilization * row.cycles
        all_correct = all_correct and row.correct
    lines.append("-" * len(header))
    avg_util = (total_weighted_util / total_cycles * 100) if total_cycles else 0.0
    lines.append(
        f"{'TOTAL':>22} {total_cycles:>10} {avg_util:>6.1f}% "
        f"{'PASS' if all_correct else 'FAIL':>5}"
    )
    return "\n".join(lines)


def format_markdown_table(app_name: str, rows: list[BenchRow]) -> str:
    lines = [
        f"# Benchmark: {app_name}",
        "",
        "| config | cycles | mult util % | correct |",
        "|---|---:|---:|:---:|",
    ]
    total_cycles = 0
    total_weighted_util = 0.0
    all_correct = True
    for row in rows:
        lines.append(
            f"| {row.label} | {row.cycles} | {row.mult_utilization * 100:.1f}% "
            f"| {'✓' if row.correct else '✗'} |"
        )
        total_cycles += row.cycles
        total_weighted_util += row.mult_utilization * row.cycles
        all_correct = all_correct and row.correct
    avg_util = (total_weighted_util / total_cycles * 100) if total_cycles else 0.0
    lines.append(
        f"| **TOTAL** | {total_cycles} | {avg_util:.1f}% "
        f"| {'✓' if all_correct else '✗'} |"
    )
    lines.append("")
    return "\n".join(lines)


def print_and_write_table(
    app_name: str, rows: list[BenchRow], out_path: Path | None = None,
) -> None:
    """Print the console table and, if out_path given, write the markdown table."""
    print(format_console_table(app_name, rows))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_markdown_table(app_name, rows), encoding="utf-8")
        print(f"\nWrote {out_path}")
