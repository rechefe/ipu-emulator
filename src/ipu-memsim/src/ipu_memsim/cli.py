"""The memory what-if tool's command line.

    memsim record  KERNEL [--case NAME] [--set KEY=VALUE ...] [--format FMT] [--out FILE]
    memsim replay  TRACE --config FILE.toml [--json]
    memsim sweep   TRACE --out DIR
    memsim network NETWORK.toml --out DIR [--configs FILE.toml ...]
    memsim dram-measure --dramsim3 PROGRAM --ini CONFIG.ini [--out FILE.toml]

Run it through Bazel (paths may be relative to where you call it from):

    bazel run //src/ipu-memsim:memsim -- record residual_add
    bazel run //src/ipu-memsim:memsim -- replay memsim-out/residual_add.default.csv.gz \\
        --config src/ipu-memsim/configs/cache_unit.toml

Outputs go to ``memsim-out/`` at the top of the checkout unless ``--out``
says otherwise; that folder is in ``.gitignore``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from ipu_memsim.config import load_config, make_model
from ipu_memsim.replay import ReplayResult, replay
from ipu_memsim.trace import Trace, load_trace, save_trace
from ipu_memsim.workset import working_set

__all__ = ["main"]


def _workspace() -> Path:
    """The checkout's top folder (``bazel run`` sets it), else the current one."""
    return Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))


def _user_path(text: str) -> Path:
    """A path as the user typed it: relative to where they ran the command."""
    return Path(os.environ.get("BUILD_WORKING_DIRECTORY", ".")) / text


def _commit() -> str:
    """The checkout's git commit, or ``"unknown"`` without git."""
    try:
        done = subprocess.run(
            ["git", "-C", str(_workspace()), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return done.stdout.strip()


def _value(text: str) -> object:
    """A ``--set`` value: an int, a float, true/false, else the text."""
    if text in ("true", "false"):
        return text == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def describe(trace: Trace) -> str:
    """A trace's summary: the run, and each region's working set."""
    header = trace.header
    lines = [
        f"{header['kernel']} / {header['case']} / {header['format']}: "
        f"{trace.ideal_cycles} cycles, {len(trace)} XMEM accesses "
        f"(emulator stats: {header['stats']['xmem_reads']} reads, "
        f"{header['stats']['xmem_writes']} writes; match = {header['stats_match']})",
        f"{'region':<14}{'role':<9}{'rows':>8}{'reads':>9}{'writes':>9}"
        f"{'footprint':>11}{'peak live':>11}{'look-back':>11}",
    ]
    for w in working_set(trace):
        lines.append(
            f"{w.name:<14}{w.role:<9}{w.rows:>8}{w.reads:>9}{w.writes:>9}"
            f"{w.footprint:>11}{w.peak_live:>11}{w.look_back:>11}"
        )
    return "\n".join(lines)


def describe_result(result: ReplayResult) -> str:
    """A replay's summary: totals, then one line per region."""
    lines = [
        f"model {result.model}: ideal {result.ideal_cycles} + stall {result.stall_cycles} "
        f"= {result.total_cycles} cycles (x{result.slowdown:.3f})",
        f"{'region':<14}{'reads':>9}{'writes':>9}{'stalled':>9}{'stall cycles':>14}",
    ]
    for name, stats in result.regions.items():
        lines.append(
            f"{name:<14}{stats.reads:>9}{stats.writes:>9}{stats.stalled:>9}{stats.stall_cycles:>14}"
        )
    counters = {k: v for k, v in result.counters.items() if k != "tables"}
    if counters:
        lines.append("counters: " + ", ".join(f"{k}={v}" for k, v in counters.items()))
    for name, table in result.counters.get("tables", {}).items():
        lines.append(f"  table {name}: " + ", ".join(f"{k}={v}" for k, v in table.items()))
    return "\n".join(lines)


def _record(args: argparse.Namespace) -> int:
    from ipu_memsim.record import record  # imports the harness: only when recording

    values = dict(item.split("=", 1) for item in args.set)
    values = {key: _value(text) for key, text in values.items()}
    started = time.perf_counter()
    trace = record(args.kernel, args.case, values, args.format, commit=_commit())
    label = args.case if not values else "_".join(f"{k}{v}" for k, v in values.items())
    out = (
        _user_path(args.out)
        if args.out
        else _workspace() / "memsim-out" / f"{args.kernel}.{label}.csv.gz"
    )
    save_trace(trace, out)
    print(describe(trace))
    print(f"recorded in {time.perf_counter() - started:.1f} s -> {out}")
    return 0 if trace.header["stats_match"] else 1


def _replay(args: argparse.Namespace) -> int:
    trace = load_trace(_user_path(args.trace))
    result = replay(trace, make_model(load_config(_user_path(args.config))))
    if args.json:
        print(json.dumps(asdict(result) | {"total_cycles": result.total_cycles}, indent=2))
    else:
        print(describe(trace))
        print(describe_result(result))
    return 0


def _sweep(args: argparse.Namespace) -> int:
    from ipu_memsim.sweep import run_sweeps

    for path in run_sweeps(load_trace(_user_path(args.trace)), _user_path(args.out)):
        print(path)
    return 0


def _network(args: argparse.Namespace) -> int:
    from ipu_memsim.network import run_network

    configs = [_user_path(c) for c in args.configs]
    for path in run_network(
        _user_path(args.network), _user_path(args.out), configs, commit=_commit()
    ):
        print(path)
    return 0


def _dram_measure(args: argparse.Namespace) -> int:
    from ipu_memsim.dram_trace import measure_fill_table

    ini = _user_path(args.ini)
    table = measure_fill_table(_user_path(args.dramsim3), ini, ipu_clock_ns=args.ipu_clock_ns)
    lines = [
        f"# Measured with DRAMsim3 on {ini.name}: IPU cycles ({args.ipu_clock_ns} ns each) to move",
        "# N consecutive 128-byte rows in one transfer. Generated by `memsim dram-measure`.",
        "[model]",
        'kind = "cache_unit"',
        f"num_banks = {args.num_banks}",
        f"bank_rows = {args.bank_rows}",
        "",
        "[dram]",
        'kind = "table"',
    ]
    for direction in ("fill", "drain"):
        cells = ", ".join(f'"{rows}" = {cycles}' for rows, cycles in table[direction].items())
        lines.append(f"{direction} = {{ {cells} }}")
    text = "\n".join(lines) + "\n"
    if args.out:
        _user_path(args.out).write_text(text)
    print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memsim", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    rec = commands.add_parser("record", help="run a kernel once and save its XMEM trace")
    rec.add_argument("kernel", help="a registered kernel name, e.g. residual_add")
    rec.add_argument("--case", default="default", help="the kernel's case (default: default)")
    rec.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="a parameter value over the case's (repeatable)")  # fmt: skip
    rec.add_argument("--format", default=None, help="element format (default: the case's first)")
    rec.add_argument("--out", default=None, help="trace file (.csv or .csv.gz)")
    rec.set_defaults(run=_record)

    rep = commands.add_parser("replay", help="replay a trace through a memory model")
    rep.add_argument("trace")
    rep.add_argument("--config", required=True, help="a model config (TOML)")
    rep.add_argument("--json", action="store_true", help="print the result as JSON")
    rep.set_defaults(run=_replay)

    swp = commands.add_parser("sweep", help="sweep bank count, XMEM size and miss penalty")
    swp.add_argument("trace")
    swp.add_argument("--out", required=True, help="folder for the CSV and SVG files")
    swp.set_defaults(run=_sweep)

    net = commands.add_parser("network", help="per-layer-type table of a network")
    net.add_argument("network", help="a network description (TOML)")
    net.add_argument("--out", required=True, help="folder for traces and the table")
    net.add_argument("--configs", nargs="*", default=[], help="model configs to compare")
    net.set_defaults(run=_network)

    dram = commands.add_parser("dram-measure", help="measure bank-fill times with DRAMsim3")
    dram.add_argument("--dramsim3", required=True, help="the dramsim3main.out program")
    dram.add_argument(
        "--ini", required=True, help="a DRAMsim3 configuration, e.g. HBM2_8Gb_x128.ini"
    )
    dram.add_argument(
        "--ipu-clock-ns", type=float, default=1.0, help="IPU clock period (default 1 ns)"
    )
    dram.add_argument("--num-banks", type=int, default=16)
    dram.add_argument("--bank-rows", type=int, default=1024)
    dram.add_argument("--out", default=None, help="write the config here (TOML)")
    dram.set_defaults(run=_dram_measure)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
