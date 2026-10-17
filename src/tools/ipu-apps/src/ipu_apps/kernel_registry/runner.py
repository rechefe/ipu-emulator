"""Thin command-line frontend for registry cases, shared by every kernel."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ipu_apps.kernel_registry.cases import KernelCase, case_options, load_cases, run_case
from ipu_isa.errors import IpuError


def _user_path(text: str) -> Path:
    """A path the user typed: relative to where they ran ``bazel run``.

    ``bazel run`` executes inside the runfiles tree and names the caller's
    directory in ``BUILD_WORKING_DIRECTORY``; without this, a relative
    ``--output`` would be written somewhere under ``bazel-bin``.
    """
    path = Path(text)
    base = os.environ.get("BUILD_WORKING_DIRECTORY")
    return Path(base) / path if base and not path.is_absolute() else path


def _selector() -> argparse.ArgumentParser:
    """The options that pick what runs; parsed first, to find the case's options."""
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--kernel", help="registered kernel name")
    selector.add_argument("--case", default="default", help="named case (default: default)")
    selector.add_argument("--list-cases", action="store_true", help="list cases without running")
    return selector


def build_parser(case: KernelCase | None = None) -> argparse.ArgumentParser:
    """The runner's command line: the shared options, then each of ``case``'s options.

    Every kernel target passes ``--kernel``; the docs' CLI reference renders
    this parser, and each case option is ``case_options(case)``'s flag.
    """
    parser = argparse.ArgumentParser(description=__doc__, parents=[_selector()], allow_abbrev=False)
    parser.add_argument(
        "--max-cycles", type=int, help="the cycle limit for this run (default: the case's)"
    )
    parser.add_argument(
        "--profile-aliases", action="store_true", help="collect full ISA alias measurements"
    )
    parser.add_argument(
        "--alias-report",
        type=_user_path,
        help="write full ISA measurements as JSON (enables profiling)",
    )
    parser.add_argument(
        "--output",
        type=_user_path,
        help="export completed output, including output that fails validation",
    )
    for option in case_options(case).values() if case is not None else ():
        default = option["default"]
        how = (
            {"action": argparse.BooleanOptionalAction}
            if isinstance(default, bool)
            else {"type": type(default)}
        )
        parser.add_argument(option["flag"], default=default, **how)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = _selector()
    parser = build_parser()
    selected, _ = selector.parse_known_args(argv)
    if selected.kernel is None:
        # argparse handles generic --help before checking the kernel requirement.
        parser.parse_args(argv)
        parser.error("the following arguments are required: --kernel")
    try:
        cases = load_cases(selected.kernel)
        if selected.list_cases:
            parser.parse_args(argv)
            print("\n".join(cases))
            return 0
        if selected.case not in cases:
            raise ValueError(f"unknown case {selected.case!r}; available: {', '.join(cases)}")
        case = cases[selected.case]
        parser = build_parser(case)
        args = parser.parse_args(argv)
        from ipu_emu.alias_profile import AliasProfile

        profile = (
            AliasProfile(metadata={"case": args.case})
            if args.profile_aliases or args.alias_report
            else None
        )
        state, cycles = run_case(
            args.kernel,
            case,
            options={k: getattr(args, k) for k in case.defaults},
            max_cycles=args.max_cycles,
            output_path=args.output,
            alias_profile=profile,
        )
        if state.quit_requested:
            # The user quit the debugger: an unfinished run has no result.
            return 0
        if args.alias_report:
            args.alias_report.write_text(profile.to_json(indent=2) + "\n")
    except (
        IpuError,
        ValueError,
        OSError,
        RuntimeError,
        AssertionError,
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        argparse.ArgumentError,
    ) as exc:
        parser.exit(1, f"error: {str(exc) or type(exc).__name__}\n")
    print(f"{args.kernel}/{args.case}: PASS ({cycles} cycles)")
    print(state.stats.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
