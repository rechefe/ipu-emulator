"""Thin command-line frontend for registry cases, shared by every kernel."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ipu_apps.kernel_registry.cases import load_cases, run_case


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--kernel", help="registered kernel name")
    selector.add_argument("--case", default="default", help="named case (default: default)")
    selector.add_argument("--list-cases", action="store_true", help="list cases without running")
    parser = argparse.ArgumentParser(description=__doc__, parents=[selector], allow_abbrev=False)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--output", type=Path,
                        help="export completed output, including output that fails validation")
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
        for name, default in case.defaults.items():
            option = "--" + name.replace("_", "-")
            if isinstance(default, bool):
                parser.add_argument(option, default=default, action=argparse.BooleanOptionalAction)
            else:
                parser.add_argument(option, type=type(default), default=default)
        args = parser.parse_args(argv)
        state, cycles = run_case(args.kernel, case,
                                options={k: getattr(args, k) for k in case.defaults},
                                max_cycles=args.max_cycles, output_path=args.output)
    except (ValueError, OSError, RuntimeError, AssertionError, ImportError,
            AttributeError, KeyError, TypeError, argparse.ArgumentError) as exc:
        parser.exit(1, f"error: {str(exc) or type(exc).__name__}\n")
    print(f"{args.kernel}/{args.case}: PASS ({cycles} cycles)")
    print(state.stats.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
