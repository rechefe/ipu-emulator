"""Ask the registry which kernel handles a computation, for any operation.

Usage::

    bazel run //src/tools/ipu-apps:query                                # coverage report
    bazel run //src/tools/ipu-apps:query -- softmax shape=32,300 dim=1  # which kernel, and why
    bazel run //src/tools/ipu-apps:query -- softmax shape=8,n dim=1 --sweep n=1..200

Parameters are ``NAME=VALUE`` and are passed to :func:`resolve` verbatim. A
value containing a comma is a tuple of integers (``shape=300,`` is 1-D), an
integer is an int, ``true``/``false`` a bool, anything else a string. ``--sweep NAME=START..END``
substitutes each integer for ``NAME`` in the one parameter that mentions it and
prints where the winning kernel changes (:func:`boundaries`).

Exits 0 when the query is covered (for a sweep: every value), 1 when not.
"""
from __future__ import annotations

import argparse
import sys

from ipu_apps.kernel_registry.coverage import boundaries, report
from ipu_apps.kernel_registry.registry import resolve


def _tokens(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _value(text: str, sweep: str | None = None, at: int | None = None):
    """Parse one parameter value, substituting ``at`` for the ``sweep`` token."""
    def scalar(token: str):
        if token == sweep:
            return at
        if token.lower() in ("true", "false"):
            # Left as a string, "False" would be truthy.
            return token.lower() == "true"
        try:
            return int(token)
        except ValueError:
            return token

    if "," in text:
        return tuple(scalar(token) for token in _tokens(text))
    return scalar(text.strip())


def _sweep(op: str, params: dict[str, str], spec: str) -> int:
    name, _, span = spec.partition("=")
    start, dots, end = span.partition("..")
    if not (name and dots and start.isdigit() and end.isdigit()):
        raise ValueError(f"--sweep takes NAME=START..END, got {spec!r}")
    if int(start) > int(end):
        raise ValueError(f"--sweep {spec!r}: START must not exceed END")
    swept = [param for param, text in params.items() if name in _tokens(text)]
    if len(swept) != 1:
        raise ValueError(
            f"--sweep {name}: exactly one parameter must mention {name!r}, "
            f"found {len(swept)}"
        )
    template = params.pop(swept[0])
    runs = boundaries(
        op, swept[0], range(int(start), int(end) + 1),
        build=lambda at: _value(template, name, at),
        **{param: _value(text) for param, text in params.items()},
    )
    for run in runs:
        print(run.render(name))
    return 0 if all(run.kernel for run in runs) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("op", nargs="?",
                        help="operation, e.g. softmax (omit for the coverage report)")
    parser.add_argument("params", nargs="*", metavar="NAME=VALUE",
                        help="query parameters, e.g. shape=32,300 dim=1")
    parser.add_argument("--sweep", metavar="NAME=START..END",
                        help="print the routing table as NAME runs over START..END")
    args = parser.parse_args(argv)

    if args.op is None:
        print(report())
        return 0

    params: dict[str, str] = {}
    for item in args.params:
        name, sep, text = item.partition("=")
        if not (name and sep and text.strip()):
            parser.error(f"expected NAME=VALUE, got {item!r}")
        params[name] = text

    try:
        if args.sweep:
            return _sweep(args.op, params, args.sweep)
        verdict = resolve(args.op, **{name: _value(text) for name, text in params.items()})
    except ValueError as exc:
        parser.error(str(exc))

    print(verdict.describe())
    return 0 if verdict.supported else 1


if __name__ == "__main__":
    sys.exit(main())
