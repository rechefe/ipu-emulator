#!/usr/bin/env python3
"""Run the fully-connected layer with the interactive debug CLI.

Intended to be run via ``bazel run`` so paths to the assembled binary
and test data are resolved through runfiles automatically.

Usage:
    bazel run //src/tools/ipu-emu-py:run_fc_debug -- --dtype INT8

The assembly already contains ``break.if_eq lr5 0`` which fires on
every outer-loop iteration.  When the debugger stops, type ``help``
to see available commands.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _runfiles_dir() -> Path | None:
    # Method 1: explicit env vars (py_test sets TEST_SRCDIR)
    srcdir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    if srcdir:
        return Path(srcdir) / workspace

    # Method 2: extract from PYTHONPATH (Bazel py_binary puts runfiles here)
    pp = os.environ.get("PYTHONPATH", "")
    for entry in pp.split(os.pathsep):
        if entry.endswith(os.sep + "_main") and ".runfiles" in entry:
            return Path(entry)

    return None


def _resolve(*parts: str) -> Path:
    rf = _runfiles_dir()
    if rf:
        return rf.joinpath(*parts)
    # Fallback: workspace root (find MODULE.bazel)
    for parent in Path(__file__).resolve().parents:
        if (parent / "MODULE.bazel").exists():
            return parent.joinpath(*parts)
    raise FileNotFoundError("Cannot determine workspace root")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fully_connected with interactive debug CLI."
    )
    parser.add_argument(
        "--dtype",
        default="INT8",
        choices=["INT8", "FP8_E4M3", "FP8_E5M2"],
        help="Data type (default: INT8)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=2_000_000,
        help="Safety limit for max cycles (default: 2000000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write output binary (optional)",
    )
    args = parser.parse_args()

    dtype_dir = {
        "INT8": "int8",
        "FP8_E4M3": "fp8_e4m3",
        "FP8_E5M2": "fp8_e5m2",
    }[args.dtype]

    inst_path = _resolve(
        "src", "tools", "ipu-emu-py", "assemble_fully_connected.bin"
    )
    data_dir = _resolve(
        "src", "apps", "fully_connected", "test_data_format", dtype_dir
    )
    inputs_path = data_dir / f"inputs_{dtype_dir}.bin"
    weights_path = data_dir / f"weights_{dtype_dir}.bin"

    for p in [inst_path, inputs_path, weights_path]:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Import here so Bazel resolves deps first
    from ipu_emu.debug_cli import debug_prompt
    from ipu_emu.apps.fully_connected import run_fully_connected

    print(f"Running fully_connected ({args.dtype}) with debug CLI")
    print(f"  Instructions : {inst_path}")
    print(f"  Inputs       : {inputs_path}")
    print(f"  Weights      : {weights_path}")
    print(f"  Max cycles   : {args.max_cycles}")
    print()
    print("The program will stop at each breakpoint.")
    print("Type 'help' for available commands, 'continue' to resume, 'quit' to exit.")
    print()

    state, cycles = run_fully_connected(
        inst_path=inst_path,
        inputs_path=inputs_path,
        weights_path=weights_path,
        output_path=args.output,
        dtype=args.dtype,
        max_cycles=args.max_cycles,
        debug_callback=lambda s, c: debug_prompt(s),
    )

    print(f"\nExecution finished after {cycles} cycles.")


if __name__ == "__main__":
    main()
