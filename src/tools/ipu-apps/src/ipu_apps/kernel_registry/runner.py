"""Thin command-line frontend for registry cases, shared by every kernel."""
from __future__ import annotations

import argparse
from pathlib import Path

from ipu_apps.kernel_registry.cases import load_cases, run_case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--case", default="default")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--output", type=Path)
    # Select the case before adding its options, so --help includes them.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--kernel", required=True)
    selector.add_argument("--case", default="default")
    selected, _ = selector.parse_known_args(argv)
    try:
        cases = load_cases(selected.kernel)
        if selected.case not in cases:
            raise ValueError(f"unknown case {selected.case!r}; available: {', '.join(cases)}")
        case = cases[selected.case]
        for name, default in case.defaults.items():
            if isinstance(default, bool):
                parser.add_argument("--" + name.replace("_", "-"), default=default,
                                    action=argparse.BooleanOptionalAction)
            else:
                parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
        args = parser.parse_args(argv)
        if args.list_cases:
            print("\n".join(cases))
            return 0
        state, cycles = run_case(args.kernel, case,
                                options={k: getattr(args, k) for k in case.defaults},
                                max_cycles=args.max_cycles, output_path=args.output)
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"{args.kernel}/{args.case}: PASS ({cycles} cycles)")
    print(state.stats.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
