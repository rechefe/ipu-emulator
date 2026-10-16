"""Summarise a combined lcov report as a Markdown table, one row per package.

Used by the nightly workflow (.github/workflows/nightly.yml) to write the
coverage of `bazel coverage --config=full //...` into the job summary:

    bazel run //tools:coverage_summary -- \
        "$PWD/bazel-out/_coverage/_coverage_report.dat" \
        --require ipu_emu --require ipu_as --require ipu_isa

A source file ``src/tools/<dir>/src/<module>/...`` counts towards ``<module>``;
any other file counts towards its directory (``docs``, ``tools``, ...).

Exits non-zero when the report is missing or empty, when it measures no line
at all, or when a ``--require``d package has no covered line, so a broken
coverage setup fails the job instead of uploading an empty report.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass
class Counts:
    hit: int = 0
    found: int = 0

    def add(self, other: Counts) -> None:
        self.hit += other.hit
        self.found += other.found

    @property
    def percent(self) -> float:
        return 100.0 * self.hit / self.found if self.found else 0.0


def package_of(source: str) -> str:
    """The package a source file path from an lcov ``SF:`` record belongs to."""
    parts = PurePosixPath(source).parts
    # Absolute paths (some coverage tools record them): keep the repo-relative tail.
    for i in range(len(parts) - 1):
        if parts[i : i + 2] == ("src", "tools"):
            parts = parts[i:]
            break
    if len(parts) >= 6 and parts[:2] == ("src", "tools") and parts[3] == "src":
        return parts[4]
    if len(parts) >= 4 and parts[:2] == ("src", "tools"):
        return "/".join(parts[:3])
    return parts[0] if len(parts) > 1 else "(root)"


def parse_lcov(text: str) -> dict[str, Counts]:
    """Line counts per source file, from ``DA:`` records (``LH``/``LF`` as fallback).

    A file reported by several tests appears in several records; a line counts
    as hit if any record hit it.
    """
    lines: dict[str, dict[int, bool]] = {}
    totals: dict[str, Counts] = {}
    source: str | None = None
    lh = lf = 0
    saw_da = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            source = line[3:]
            lines.setdefault(source, {})
            lh = lf = 0
            saw_da = False
        elif line.startswith("DA:") and source is not None:
            fields = line[3:].split(",")
            number, count = int(fields[0]), int(fields[1])
            seen = lines[source]
            seen[number] = seen.get(number, False) or count > 0
            saw_da = True
        elif line.startswith("LH:"):
            lh = int(line[3:])
        elif line.startswith("LF:"):
            lf = int(line[3:])
        elif line == "end_of_record" and source is not None:
            if not saw_da and lf:
                totals.setdefault(source, Counts()).add(Counts(lh, lf))
            source = None
    result: dict[str, Counts] = {}
    for src, seen in lines.items():
        counts = Counts(sum(seen.values()), len(seen))
        if src in totals:
            counts.add(totals[src])
        result[src] = counts
    return result


def summarise(per_file: dict[str, Counts]) -> dict[str, Counts]:
    per_package: dict[str, Counts] = {}
    for src, counts in per_file.items():
        per_package.setdefault(package_of(src), Counts()).add(counts)
    return per_package


def markdown(per_package: dict[str, Counts]) -> str:
    total = Counts()
    rows = ["| Package | Lines hit | Lines found | Line coverage |", "|---|---:|---:|---:|"]
    for name in sorted(per_package):
        c = per_package[name]
        total.add(c)
        rows.append(f"| `{name}` | {c.hit} | {c.found} | {c.percent:.1f} % |")
    rows.append(f"| **Total** | {total.hit} | {total.found} | {total.percent:.1f} % |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="combined lcov report (.dat)")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="fail unless PACKAGE has at least one covered line (repeatable)",
    )
    args = parser.parse_args(argv)

    if not args.report.is_file() or args.report.stat().st_size == 0:
        print(f"error: coverage report {args.report} is missing or empty", file=sys.stderr)
        return 1
    per_package = summarise(parse_lcov(args.report.read_text(encoding="utf-8")))
    print("## Line coverage\n")
    print(markdown(per_package))

    errors = []
    if sum(c.found for c in per_package.values()) == 0:
        errors.append("the report measures no lines")
    for name in args.require:
        if per_package.get(name, Counts()).hit == 0:
            errors.append(f"package {name} has no covered line")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
