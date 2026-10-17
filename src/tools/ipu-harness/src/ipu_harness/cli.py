"""Run a kernel's cases from the command line (E06-11).

    python -m ipu_harness.cli KERNEL [--case NAME] [--format FMT|all]
        [--input NAME=PATH.npy ...] [--output DIR] [--trace PATH [--context PATH]]
        [--max-cycles N] [--list-cases] [--debug] [--profile-aliases]
        [--alias-report PATH]
    python -m ipu_harness.cli KERNEL --benchmark [--format FMT|all]
        [--markdown PATH] [--json PATH]

KERNEL is a kernel folder, or ``<family>/registry.toml::<name>`` for a family
entry (E06-17).

``--debug``, or ``IPU_DEBUG_TUI=1`` in the environment (the spelling a
``bazel run`` of a test target passes), stops the run in the debugger TUI;
quitting it (or Ctrl-C) ends the command with exit code 0 and no result.
``--profile-aliases`` prints the alias profile's summary; ``--alias-report``
writes the profile as JSON instead (one file per format with ``--format all``).
``--context`` records the run for the Cycle Inspector
(:mod:`ipu_harness.inspector`): the trace, and beside it the context a
viewer needs. This module is the only one in ``ipu_harness`` that reads the
environment.

One line per run, from :meth:`RunResult.describe`. ``--benchmark`` instead
runs the kernel's ``[benchmark]`` points (:mod:`ipu_harness.benchmark`) and
prints their table, writing it as Markdown and the rows as JSON when asked; a
point that does not pass fails the command. Exit codes:

* 0: every run passed.
* 1: a run failed or faulted, or the request was wrong: a ``ConfigurationError``,
  an assembler error in the kernel's program, an ``EmulatorError`` the model
  raised, or an ``OSError`` (one line, no traceback).
* 2: anything else, which is a bug in the harness or the kernel's Python
  (the traceback is printed). A configuration error caused by another
  exception (a generator, a pack transform or a reference that raised,
  or failed to import, torch included) counts as a bug too.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_harness import benchmark
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, expand
from ipu_harness.debug import DebugLaunch
from ipu_harness.family import ENTRY_SEPARATOR, load_kernel, split_ref
from ipu_harness.inspector import InspectorRecorder
from ipu_harness.manifest import KernelManifest
from ipu_harness.runner import RunOptions, RunResult, run_kernel
from ipu_isa.errors import IpuError
from ipu_isa.formats import ElementFormat

#: The value of ``--format`` that runs every format the case runs in.
ALL = "all"


def _caller(path: Path) -> Path:
    """``path`` from where the user ran the command (``bazel run`` changes directory)."""
    if path.is_absolute():
        return path
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / path


def _kernel(text: str) -> str:
    """A kernel folder, or ``<registry.toml>::<name>``, from where the user ran."""
    path, name = split_ref(text)
    return str(_caller(path)) + ("" if name is None else ENTRY_SEPARATOR + name)


def _input(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--input takes NAME=PATH.npy, not {text!r}")
    return name, _caller(Path(path))


def _cycles(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"--max-cycles takes 0 or more, not {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The command line in the module docstring; the docs' CLI reference renders it."""
    parser = argparse.ArgumentParser(prog="ipu_harness.cli", description=__doc__.splitlines()[0])
    parser.add_argument(
        "kernel",
        type=_kernel,
        help="the kernel's folder (it holds kernel.toml), or REGISTRY.toml::NAME for an entry",
    )
    parser.add_argument("--case", default="default", help="a case name, e.g. 'sweep[rows=7]'")
    parser.add_argument(
        "--format",
        help="an element format, or 'all' (default: the first of the case's formats)",
    )
    parser.add_argument(
        "--input",
        type=_input,
        action="append",
        default=[],
        metavar="NAME=PATH.npy",
        help="replace an input (repeatable)",
    )
    parser.add_argument("--output", type=Path, help="write each output as DIR/<name>.npy")
    parser.add_argument("--trace", type=Path, help="write the run's trace (JSON Lines)")
    parser.add_argument(
        "--context",
        type=Path,
        help="with --trace, also write the Cycle Inspector's context (JSON): the source, "
        "the starting registers, and each word's lanes and XMEM reads",
    )
    parser.add_argument(
        "--max-cycles", type=_cycles, help="the cycle limit (0: no limit; default: the model's)"
    )
    parser.add_argument("--list-cases", action="store_true", help="list the cases and stop")
    parser.add_argument(
        "--benchmark", action="store_true", help="run the [benchmark] points and print their table"
    )
    parser.add_argument("--markdown", type=Path, help="with --benchmark, write the table here")
    parser.add_argument("--json", type=Path, help="with --benchmark, write the rows as JSON here")
    parser.add_argument("--debug", action="store_true", help="stop in the debugger TUI")
    parser.add_argument(
        "--profile-aliases",
        action="store_true",
        help="profile the ISA aliases of the run and print the summary",
    )
    parser.add_argument(
        "--alias-report",
        type=Path,
        help="write the alias profile as JSON instead (implies profiling; "
        "with --format all, PATH.<format>.json each)",
    )
    return parser


#: The environment's spelling of ``--debug``.
DEBUG_ENV = "IPU_DEBUG_TUI"


def debug_launch(flag: bool, environ: Mapping[str, str]) -> DebugLaunch | None:
    """The debug launch ``--debug`` or ``IPU_DEBUG_TUI=1`` asks for, if any."""
    return DebugLaunch() if flag or environ.get(DEBUG_ENV) == "1" else None


@contextmanager
def _terminal(debugging: bool) -> Iterator[None]:
    """Hand the terminal to the debugger under ``bazel run`` of a test target.

    Bazel's test-setup.sh tees stdout into a pipe and starts its child in a
    background process group; curses needs the terminal on stdout and the
    foreground. Restored on completion, failure and cancellation.
    """
    if not debugging or not sys.stdin.isatty() or sys.stdout.isatty():
        yield
        return
    sys.stdout.flush()
    output_fd = sys.stdout.fileno()
    input_fd = sys.stdin.fileno()
    foreground = os.tcgetpgrp(input_fd)
    saved = os.dup(output_fd)
    old_signal = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
    try:
        os.tcsetpgrp(input_fd, os.getpgrp())
        os.dup2(input_fd, output_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, output_fd)
        os.close(saved)
        os.tcsetpgrp(input_fd, foreground)
        signal.signal(signal.SIGTTOU, old_signal)


def _case(manifest: KernelManifest, name: str) -> Case:
    cases = {case.name: case for case in expand(manifest)}
    if name not in cases:
        raise ConfigurationError(
            f"{manifest.name} has no case {name!r}; its cases: {', '.join(cases)}"
        )
    return cases[name]


def _formats(case: Case, requested: str | None) -> tuple[ElementFormat, ...]:
    if requested is None:
        return case.formats[:1]
    if requested == ALL:
        return case.formats
    try:
        fmt = ElementFormat(requested.lower())
    except ValueError:
        known = ", ".join(f.value for f in ElementFormat)
        raise ConfigurationError(f"--format {requested!r} is no element format ({known})") from None
    return (fmt,)


def _options(
    args: argparse.Namespace, fmt: ElementFormat, many: bool, debug: DebugLaunch | None
) -> RunOptions:
    """The run options of one format: with several, one output folder and trace each."""
    output = None
    if args.output is not None:
        output = _caller(args.output) / fmt.value if many else _caller(args.output)
    trace = None
    if args.trace is not None:
        trace = _per_format(_caller(args.trace), fmt, many)
    recorder = None
    if args.context is not None and trace is not None:
        recorder, trace = InspectorRecorder(trace), None  # the recorder writes the trace
    return RunOptions(
        max_cycles=args.max_cycles,
        inputs=dict(args.input),
        output_dir=output,
        trace=trace,
        debug=debug,
        executor=recorder,
        alias_profile=args.profile_aliases or args.alias_report is not None,
    )


def _per_format(path: Path, fmt: ElementFormat, many: bool) -> Path:
    """``path``, or ``path.<format><suffix>`` when several formats run."""
    return path.with_name(f"{path.stem}.{fmt.value}{path.suffix}") if many else path


def _report_profile(args: argparse.Namespace, result: RunResult, many: bool) -> None:
    profile: Any = result.alias_profile  # ipu_emu's AliasProfile is untyped
    if profile is None:
        return
    if args.alias_report is None:
        print(profile.format_summary())
        return
    report = _per_format(_caller(args.alias_report), result.format, many)
    report.write_text(profile.to_json(indent=2) + "\n")


def _run(args: argparse.Namespace) -> int:
    _check_options(args)
    manifest = load_kernel(args.kernel)
    if args.list_cases:
        for case in expand(manifest):
            print(f"{case.name}  ({', '.join(f.value for f in case.formats)})")
        return 0
    if args.benchmark:
        return _benchmark(manifest, args)
    case = _case(manifest, args.case)
    debug = debug_launch(args.debug, os.environ)
    formats = _formats(case, args.format)
    if args.context is not None and debug is not None:
        raise ConfigurationError("--context records the run as it is; it takes no --debug")
    passed = True
    cache = AssemblyCache()
    for fmt in formats:
        options = _options(args, fmt, len(formats) > 1, debug)
        with _terminal(debug is not None):
            result = run_kernel(manifest, case, fmt, options, cache=cache)
        print(result.describe())
        if isinstance(options.executor, InspectorRecorder):
            context = _per_format(_caller(args.context), fmt, len(formats) > 1)
            document = options.executor.context(manifest, case)
            context.write_text(json.dumps(document, separators=(",", ":")) + "\n")
        if result.quit:  # the user quit the debugger: no result, and no more runs
            return 0
        _report_profile(args, result, len(formats) > 1)
        passed = passed and result.passed
    return 0 if passed else 1


#: The options a --benchmark run takes from its [benchmark] table instead, by flag.
_RUN_ONLY: Mapping[str, Callable[[argparse.Namespace], bool]] = {
    "--case": lambda a: a.case != "default",
    "--input": lambda a: bool(a.input),
    "--output": lambda a: a.output is not None,
    "--trace": lambda a: a.trace is not None,
    "--context": lambda a: a.context is not None,
    "--max-cycles": lambda a: a.max_cycles is not None,
    "--debug": lambda a: a.debug,
    "--profile-aliases": lambda a: a.profile_aliases,
    "--alias-report": lambda a: a.alias_report is not None,
}


def _check_options(args: argparse.Namespace) -> None:
    """Refuse a flag the request would ignore."""
    if args.benchmark:
        given = [flag for flag, set_ in _RUN_ONLY.items() if set_(args)]
        if given:
            raise ConfigurationError(
                f"--benchmark runs the [benchmark] points as declared; it takes no {given[0]}"
            )
    else:
        given = [f for f, v in (("--markdown", args.markdown), ("--json", args.json)) if v]
        if given:
            raise ConfigurationError(f"{given[0]} goes with --benchmark")
        if args.context is not None and args.trace is None:
            raise ConfigurationError("--context goes with --trace")


def _benchmark(manifest: KernelManifest, args: argparse.Namespace) -> int:
    cases = benchmark.point_cases(manifest)  # refuses a kernel without a [benchmark]
    assert manifest.benchmark is not None
    rows: list[benchmark.BenchRow] = []
    cache = AssemblyCache()
    for fmt in _formats(cases[0], args.format):
        rows += benchmark.run_benchmark(manifest, fmt, cache=cache)
    for row in rows:
        if not row.result.passed:
            print(row.result.describe())
    table = benchmark.to_markdown(rows, manifest.benchmark.per)
    print(table)
    if args.markdown is not None:
        _caller(args.markdown).write_text(f"# {manifest.name} benchmark\n\n{table}")
    if args.json is not None:
        _caller(args.json).write_text(benchmark.to_json(rows) + "\n")
    return 0 if all(row.result.passed for row in rows) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (IpuError, OSError) as error:
        cause = error.__cause__
        if cause is not None and not isinstance(cause, (IpuError, OSError)):
            # Python code the harness ran (a generator, a pack transform, a
            # reference) raised: a bug there, so show where.
            traceback.print_exception(error)
            return 2
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception:  # a bug: show where  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
