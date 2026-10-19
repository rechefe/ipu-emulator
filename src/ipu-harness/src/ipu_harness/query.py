"""Ask the kernel registry which kernel computes an op, and why (E06-16).

Usage::

    bazel run //kernels:query                                 # coverage report
    bazel run //kernels:query -- scale rows=7 s=2.0           # which kernel, and why
    bazel run //kernels:query -- scale rows=n --sweep n=1..200
    bazel run //kernels:query -- scale rows=7 --json

It answers over the ``kernel.toml`` tree
(:func:`~ipu_harness.registry.default_registry`): the reason is the routing
decision, ``use`` the kernel's Bazel label with its parameters, and a note its
formats (IMPLEMENTATION-DECISIONS, "Routing registry"). Parameters are ``NAME=VALUE``: a value with a comma is a tuple (``shape=300,``
is 1-D), an integer is an int, a number with a point a float, ``true``/
``false`` a bool, anything else a string; ``format=e4`` limits the answer to one
element format. ``--sweep NAME=START..END`` substitutes each integer for
``NAME`` in the one parameter that mentions it and prints where the winning
kernel changes (``Boundary``, ``boundaries``).

Exits 0 when the query is covered (for a sweep: every value), 1 when not.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ipu_harness.registry import Registry, default_registry
from ipu_isa.formats import ElementFormat


class _Usage(ValueError):
    """A command line the tool cannot read (argparse reports it, exit 2)."""


def _tokens(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _scalar(token: str, sweep: str | None, at: int | None) -> Any:
    if sweep is not None and token == sweep:
        return at
    if token.lower() in ("true", "false"):
        return token.lower() == "true"  # left as a string, "false" would be truthy
    for kind in (int, float):
        try:
            return kind(token)
        except ValueError:
            pass
    return token


def _value(text: str, sweep: str | None = None, at: int | None = None) -> Any:
    """One parameter value (a tuple if it has a comma), ``at`` for the ``sweep`` token."""
    if "," in text:
        return tuple(_scalar(t, sweep, at) for t in _tokens(text))
    return _scalar(text.strip(), sweep, at)


@dataclass(frozen=True)
class Boundary:
    """A run of swept values one kernel (or none) takes."""

    start: int
    end: int
    kernel: str | None

    def render(self, param: str) -> str:
        """One line of the routing table: the span of ``param``, then the kernel."""
        span = (
            f"{param} = {self.start}"
            if self.start == self.end
            else f"{param} {self.start}..{self.end}"
        )
        return f"{span:<28} {self.kernel or '(not covered)'}"


def boundaries(
    registry: Registry, op: str, values: Sequence[int], query: Callable[[int], dict[str, Any]]
) -> tuple[Boundary, ...]:
    """Route ``op`` with ``query(value)`` at each value; collapse runs of the same winner.

    Args:
        registry: The kernels.
        op: The operation.
        values: Consecutive integers to sweep.
        query: The parameters at one value.

    Returns:
        Each run of values one kernel (or none) takes, in order.
    """
    runs: list[Boundary] = []
    for value in values:
        verdict = registry.resolve(op, **query(value))
        winner = verdict.kernel_name if verdict.supported else None
        if runs and runs[-1].kernel == winner and runs[-1].end == value - 1:
            runs[-1] = Boundary(runs[-1].start, value, winner)
        else:
            runs.append(Boundary(value, value, winner))
    return tuple(runs)


def report(registry: Registry) -> str:
    """The coverage report.

    Args:
        registry: The kernels.

    Returns:
        Markdown: every op and its kernels, and every kernel.toml that did not load.
    """
    ops = registry.operations()
    lines = ["# Kernel coverage", "", f"Operations: {', '.join(ops) if ops else '(none)'}"]
    lines += [f"Kernels:    {len(registry.manifests)}", ""]
    for op in ops:
        lines += [f"## {op}", ""]
        for m in registry.kernels(op):
            lines.append(f"- **{m.name}** -- `{m.label}`" + (f" -- {m.doc}" if m.doc else ""))
        lines.append("")
    if registry.invalid:
        # Surfaced, not swallowed: a kernel.toml that failed to load is a kernel
        # missing from coverage, which is what a coverage report exists to show.
        lines += ["## kernel.toml files that did not load", ""]
        lines += [f"- `{path}` -- {error}" for path, error in registry.invalid.items()]
        lines.append("")
    return "\n".join(lines)


def _sweep(registry: Registry, op: str, params: dict[str, str], spec: str) -> int:
    name, _, span = spec.partition("=")
    start, dots, end = span.partition("..")
    if not (name and dots and start.isdigit() and end.isdigit()):
        raise _Usage(f"--sweep takes NAME=START..END, got {spec!r}")
    if int(start) > int(end):
        raise _Usage(f"--sweep {spec!r}: START must not exceed END")
    swept = [p for p, text in params.items() if name in _tokens(text)]
    if len(swept) != 1:
        raise _Usage(
            f"--sweep {name}: exactly one parameter must mention {name!r}, found {len(swept)}"
        )
    template = params.pop(swept[0])
    fixed = {p: _value(text) for p, text in params.items()}
    runs = boundaries(
        registry,
        op,
        range(int(start), int(end) + 1),
        lambda at: {**fixed, swept[0]: _value(template, name, at)},
    )
    for run in runs:
        print(run.render(name))
    return 0 if all(run.kernel for run in runs) else 1


def build_parser() -> argparse.ArgumentParser:
    """The command line's parser (the docs' option table reads it)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "op", nargs="?", help="operation, e.g. scale (omit for the coverage report)"
    )
    parser.add_argument("params", nargs="*", metavar="NAME=VALUE", help="kernel parameters")
    parser.add_argument("--sweep", metavar="NAME=START..END", help="the routing table as NAME runs")
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON (one query)")
    return parser


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    """The command line; ``registry`` replaces the default one (for tests)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json and (args.op is None or args.sweep):
        parser.error("--json answers one query: give an operation and no --sweep")
    registry = registry or default_registry()
    if args.op is None:
        print(report(registry))
        return 0
    params: dict[str, str] = {}
    for item in args.params:
        name, sep, text = item.partition("=")
        if not (name and sep and text.strip()):
            parser.error(f"expected NAME=VALUE, got {item!r}")
        params[name] = text
    if "format" in params and params["format"].strip() not in {f.value for f in ElementFormat}:
        valid = ", ".join(f.value for f in ElementFormat)
        parser.error(f"format={params['format']}: no such element format; valid: {valid}")
    # Only the command line's own mistakes are usage errors; a kernel's broken
    # cost or a tie between kernels propagates as the bug it is.
    try:
        if args.sweep:
            return _sweep(registry, args.op, params, args.sweep)
    except _Usage as error:
        parser.error(str(error))
    verdict = registry.resolve(args.op, **{n: _value(t) for n, t in params.items()})
    print(json.dumps(verdict.to_dict(), indent=2) if args.json else verdict.describe())
    return 0 if verdict.supported else 1


if __name__ == "__main__":
    sys.exit(main())
