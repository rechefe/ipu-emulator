"""Run a kernel's cases from the command line (E06-11).

    python -m ipu_harness.cli KERNEL_DIR [--case NAME] [--format FMT|all]
        [--input NAME=PATH.npy ...] [--output DIR] [--trace PATH]
        [--max-cycles N] [--list-cases] [--debug] [--profile-aliases]
        [--alias-report PATH]

``--debug``, or ``IPU_DEBUG_TUI=1`` in the environment (the spelling a
``bazel run`` of a test target passes), stops the run in the debugger TUI.
This module is the only one in ``ipu_harness`` that reads the environment.

One line per run, from :meth:`RunResult.describe`. Exit codes:

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
import os
import signal
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from ipu_emu.errors import ConfigurationError
from ipu_harness.assembly import AssemblyCache
from ipu_harness.cases import Case, expand
from ipu_harness.debug import DebugLaunch
from ipu_harness.manifest import KernelManifest, load_manifest
from ipu_harness.runner import RunOptions, run_kernel
from ipu_isa.errors import IpuError
from ipu_isa.formats import ElementFormat

#: The value of ``--format`` that runs every format the case runs in.
ALL = "all"


def _caller(path: Path) -> Path:
    """``path`` from where the user ran the command (``bazel run`` changes directory)."""
    if path.is_absolute():
        return path
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / path


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipu_harness.cli", description=__doc__.splitlines()[0])
    parser.add_argument("kernel", type=Path, help="the kernel's folder (it holds kernel.toml)")
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
        "--max-cycles", type=_cycles, help="the cycle limit (0: no limit; default: the model's)"
    )
    parser.add_argument("--list-cases", action="store_true", help="list the cases and stop")
    parser.add_argument("--debug", action="store_true", help="stop in the debugger TUI")
    parser.add_argument(
        "--profile-aliases", action="store_true", help="profile the ISA aliases of the run"
    )
    parser.add_argument(
        "--alias-report", type=Path, help="write the alias profile as JSON (implies profiling)"
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


def _run(args: argparse.Namespace) -> int:
    manifest = load_manifest(_caller(args.kernel))
    if args.list_cases:
        for case in expand(manifest):
            print(f"{case.name}  ({', '.join(f.value for f in case.formats)})")
        return 0
    case = _case(manifest, args.case)
    debug = debug_launch(args.debug, os.environ)
    formats = _formats(case, args.format)
    many = len(formats) > 1
    passed = True
    cache = AssemblyCache()
    for fmt in formats:
        output = None
        if args.output is not None:
            output = _caller(args.output) / fmt.value if many else _caller(args.output)
        trace = None
        if args.trace is not None:
            trace = _caller(args.trace)
            if many:  # one trace per format: trace.e4.jsonl
                trace = trace.with_name(f"{trace.stem}.{fmt.value}{trace.suffix}")
        options = RunOptions(
            max_cycles=args.max_cycles,
            inputs=dict(args.input),
            output_dir=output,
            trace=trace,
            debug=debug,
            alias_profile=args.profile_aliases or args.alias_report is not None,
        )
        with _terminal(debug is not None):
            result = run_kernel(manifest, case, fmt, options, cache=cache)
        print(result.describe())
        if args.alias_report is not None and result.alias_profile is not None:
            report = _caller(args.alias_report)
            if many:
                report = report.with_name(f"{report.stem}.{fmt.value}{report.suffix}")
            report.write_text(result.alias_profile.to_json())
        passed = passed and result.passed
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """The command line. Returns the exit code."""
    args = _parser().parse_args(argv)
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
