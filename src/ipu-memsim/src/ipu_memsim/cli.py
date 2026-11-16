"""The memory what-if tool's command line.

    memsim record  KERNEL [--case NAME] [--set KEY=VALUE ...] [--format FMT] [--out FILE]
    memsim replay  TRACE --config FILE.toml [--json]
    memsim sweep   --spec EXPERIMENT.toml [--out DIR]
    memsim sweep   TRACE --out DIR
    memsim sensitivity --spec SENSITIVITY.toml [--out DIR]
    memsim knee --spec NUM_BANKS_SWEEP.toml [--out DIR]
    memsim formulas [--traces DIR] [--bank-rows N ...]
    memsim netstats NETWORK.toml [NETWORK.toml ...]
    memsim fpga-compare --measured FILE.csv --configs DIR [--fit] [--synthetic FILL_CYCLES]
    memsim params  [MODEL]
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
    if args.spec:
        return _experiment(args)
    if not args.trace or not args.out:
        raise SystemExit("memsim sweep: give --spec FILE.toml, or TRACE and --out DIR")
    from ipu_memsim.sweep import run_sweeps

    for path in run_sweeps(load_trace(_user_path(args.trace)), _user_path(args.out)):
        print(path)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    """``memsim sweep --spec``: run an experiment file and save its results."""
    from ipu_memsim import experiment

    spec = experiment.load_spec(_user_path(args.spec))
    size = experiment.grid_size(spec)
    limit = int(spec.get("report", {}).get("max_replays", experiment.DEFAULT_MAX_REPLAYS))
    workload = experiment.load_workload(spec, _workspace(), _commit())
    replays = size * sum(1 for item in workload if item.trace is not None)
    if replays > limit and not args.yes:
        raise SystemExit(f"this grid needs {replays} replays (limit {limit}); pass --yes to run it")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\r{done}/{total} replays", end="", file=sys.stderr, flush=True)

    table = experiment.run_sweep(spec, _workspace(), workload=workload, progress=progress)
    print(file=sys.stderr)
    out = _user_path(args.out) if args.out else _workspace() / "memsim-out" / table.spec["name"]
    written = experiment.write_results(table, out)
    names = [p.name for p in table.params]
    print(
        f"{'group':<16}"
        + "".join(f"{n:>22}" for n in names)
        + f"{'ideal':>12}{'stall':>12}{'stall %':>10}"
    )
    for row in table.rows:
        percent = "n/a" if row["stall_percent"] is None else f"{row['stall_percent']:.2f}"
        stall = "n/a" if row["stall_cycles"] is None else row["stall_cycles"]
        print(f"{row['group']:<16}" + "".join(f"{row[n]:>22g}" if isinstance(row[n], float) else f"{row[n]!s:>22}" for n in names)
              + f"{row['ideal_cycles']:>12}{stall!s:>12}{percent:>10}")  # fmt: skip
    if table.placeholders:
        print("TOY MODEL: placeholder values in use for "
              + ", ".join(f"{p.name} ({p.status})" for p in table.placeholders))  # fmt: skip
    print(f"{table.replays} replays in {table.seconds:.1f} s")
    for path in written:
        print(path)
    return 0


def _sensitivity(args: argparse.Namespace) -> int:
    """``memsim sensitivity --spec``: vary each ASSUMED parameter alone and rank them."""
    from ipu_memsim import experiment, sensitivity

    spec = experiment.load_spec(_user_path(args.spec))
    workload = experiment.load_workload(spec, _workspace(), _commit())
    rows = sensitivity.run_sensitivity(
        spec, _workspace(), workload=workload,
        progress=lambda name: print(f"varying {name}", file=sys.stderr),
    )  # fmt: skip
    ranking = sensitivity.rank(rows)
    # The two most sensitive parameters together, unless the spec names a pair.
    first, second = spec.get("pair", [entry["param"] for entry in ranking[:2]])
    pair = sensitivity.run_pair(spec, first, second, _workspace(), workload=workload)
    out = (
        _user_path(args.out)
        if args.out
        else _workspace() / "memsim-out" / spec.get("name", "sensitivity")
    )
    written = sensitivity.write_sensitivity(spec, rows, out, pair)
    print((out / "ranking.md").read_text())
    for path in written:
        print(path)
    return 0


def _knee(args: argparse.Namespace) -> int:
    """``memsim knee --spec``: a num_banks sweep, its knees, and the formulas' banks beside them."""
    import csv

    from ipu_memsim import experiment, formulas
    from ipu_memsim.knee import knees

    spec = experiment.load_spec(_user_path(args.spec))
    workload = experiment.load_workload(spec, _workspace(), _commit())
    table = experiment.run_sweep(spec, _workspace(), workload=workload)
    bank_rows = int(spec.get("fixed", {}).get("bank_rows", 1024))
    found = knees(table, bank_rows, float(table.spec["report"]["threshold_percent"]))
    # The closed-form prediction: the most banks any kernel of the group needs.
    for entry in found:
        needs = []
        for item in workload:
            if item.group == entry["group"] and item.trace is not None:
                tables = formulas.tables_for(
                    item.trace.header["kernel"], item.trace.header["params"]
                )
                needs.append(None if tables is None else formulas.kernel_banks(tables, bank_rows))
        covered = [n for n in needs if n is not None]
        entry["formula_banks"] = max(covered) if covered else None
        entry["kernels_without_formula"] = sum(1 for n in needs if n is None)
    out = _user_path(args.out) if args.out else _workspace() / "memsim-out" / table.spec["name"]
    written = experiment.write_results(table, out)
    with open(out / "knees.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(found[0]))
        for entry in found:
            writer.writerow(
                [
                    "" if v is None else (f"{v:.4g}" if isinstance(v, float) else v)
                    for v in entry.values()
                ]
            )
    print(
        f"{'group':<16}{'threshold knee':>16}{'plateau knee':>14}{'rows':>8}{'MB':>8}{'formula':>9}{'stall % at most banks':>23}"
    )
    for e in found:
        mb = "" if e["plateau_knee_mb"] is None else f"{e['plateau_knee_mb']:.3f}"
        final = "n/a" if e["stall_at_most_banks"] is None else f"{e['stall_at_most_banks']:.2f}"
        print(f"{e['group']:<16}{e['threshold_knee_banks']!s:>16}{e['plateau_knee_banks']!s:>14}"
              f"{e['plateau_knee_rows']!s:>8}{mb:>8}{e['formula_banks']!s:>9}{final:>23}")  # fmt: skip
    for path in [*written, out / "knees.csv"]:
        print(path)
    return 0


def _formulas(args: argparse.Namespace) -> int:
    """``memsim formulas``: check the working-set formulas against saved traces."""
    import csv
    from collections import Counter

    from ipu_memsim import formulas

    folder = _user_path(args.traces) if args.traces else _workspace() / "memsim-out" / "traces"
    rows = []
    uncovered = []
    for path in sorted(folder.glob("*.csv*")):
        trace = load_trace(path)
        checked = formulas.validate(trace, tuple(args.bank_rows))
        rows += checked
        if not checked:
            uncovered.append(trace.header["kernel"])
    out = _user_path(args.out) if args.out else _workspace() / "memsim-out" / "formulas"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "validation.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(rows[0]) if rows else ["kernel"])
        writer.writerows(list(row.values()) for row in rows)
    exact = sum(
        1
        for r in rows
        if r["rows_formula"] == r["rows_trace"] and r["look_back_formula"] == r["look_back_trace"]
    )
    print(f"{len(rows)} table checks; rows and look-back exact in {exact}")
    print("banks, formula against replay:", dict(Counter(r["verdict"] for r in rows)))
    for row in rows:
        if row["verdict"] != "match":
            print(f"  {row['verdict']}: {row['kernel']} table {row['table']} at {row['bank_rows']} rows per bank: "
                  f"formula {row['banks_formula']}, replay {row['banks_replay']}")  # fmt: skip
    print("kernels without a formula:", ", ".join(sorted(set(uncovered))) or "none")
    print(out / "validation.csv")
    return 0


def _netstats(args: argparse.Namespace) -> int:
    """``memsim netstats``: the network-only table of each network file, side by side."""
    import csv

    from ipu_memsim.netstats import network_table

    out = _user_path(args.out) if args.out else _workspace() / "memsim-out" / "netstats"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Network-only tables", "",
             "No kernel was run for these networks. Weights and activations in bytes at one byte per element.",
             "`knee (banks)` is EXTRAPOLATED from formulas validated on MobileViT-S shapes, not replayed;",
             "it is the largest value among the type's layers a formula covers (`covered`).", ""]  # fmt: skip
    with open(out / "netstats.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["network", "type", "layers", "weight_bytes", "input_bytes", "output_bytes", "operations",
                         "operations_per_byte", "knee_banks_extrapolated", "covered"])  # fmt: skip
        for path in args.networks:
            table = network_table(_user_path(path), args.bank_rows)
            published = table["published_params"]
            check = "no published count given" if not published else (
                f"published {published:,}: {100 * (table['total_params'] / published - 1):+.2f}%")  # fmt: skip
            lines += [f"## {table['name']}", "", f"Parameters counted: {table['total_params']:,} ({check}).", "",
                      "| layer type | layers | weights (bytes) | input (bytes) | output (bytes) | operations | ops per byte | knee (banks, extrapolated) | covered |",
                      "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for row in table["rows"]:
                writer.writerow([table["name"], *row.values()])
                knee = "not covered" if row["knee_banks"] is None else row["knee_banks"]
                lines.append(f"| {row['type']} | {row['layers']} | {row['weight_bytes']:,} | {row['input_bytes']:,} | "
                             f"{row['output_bytes']:,} | {row['operations']:,} | {row['operations_per_byte']:.1f} | {knee} | {row['covered']} |")  # fmt: skip
            reasons = sorted({f"{item['reason']}" for item in table["not_covered"]})
            lines += [
                "",
                f"Not covered by a formula ({len(table['not_covered'])} layer shapes): "
                + ("; ".join(reasons) or "none")
                + ".",
                "",
            ]
    (out / "netstats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(out / "netstats.md")
    print(out / "netstats.csv")
    return 0


def _fpga(args: argparse.Namespace) -> int:
    """``memsim fpga-compare``: model against measurements; with ``--fit``,
    calibrate the bank fill time on some runs and check on the others."""
    from ipu_memsim import fpga

    traces_dir = _user_path(args.traces) if args.traces else _workspace() / "memsim-out" / "traces"
    traces = {
        path.name.split(".csv")[0]: load_trace(path) for path in sorted(traces_dir.glob("*.csv*"))
    }
    configs = {
        path.stem: load_config(path) for path in sorted(_user_path(args.configs).glob("*.toml"))
    }
    if args.synthetic is not None:
        # No FPGA data yet: make labelled synthetic "measurements" from the model.
        runs = [(kernel, config) for kernel in traces for config in configs]
        measured = fpga.synthetic_measurements(runs, traces, configs, args.synthetic, args.noise)
        fpga.save_measurements(measured, _user_path(args.measured),
                               "SYNTHETIC data generated by memsim fpga-compare --synthetic; not FPGA measurements")  # fmt: skip
    measured = fpga.load_measurements(_user_path(args.measured))
    out = _user_path(args.out) if args.out else _workspace() / "memsim-out" / "fpga"
    out.mkdir(parents=True, exist_ok=True)
    fitted = None
    title = f"Model against measurements: {Path(args.measured).name}"
    if args.fit:
        # Every other run calibrates; the rest are held out for the check.
        calibration, held_out = measured[0::2], measured[1::2]
        fitted, _ = fpga.fit_fill_time(calibration, traces, configs)
        configs = {name: fpga._with_fill(config, fitted) for name, config in configs.items()}
        measured = held_out
        title += " (held-out runs, after calibration)"
    text = fpga.report(fpga.compare(measured, traces, configs), title, fitted)
    (out / "fpga_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(out / "fpga_report.md")
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


def _params(args: argparse.Namespace) -> int:
    from ipu_memsim.params import MODEL_KINDS, params_for

    for kind in [args.model] if args.model else MODEL_KINDS:
        print(f"[{kind}]")
        for p in params_for(kind):
            limits = f"{p.low:g}..{p.high:g}" if p.low is not None else "/".join(p.choices)
            print(f"  {p.name:<24}{p.status:<10}default {p.default!s:<16}{p.unit:<11}{limits}")
            print(f"  {'':<24}{p.description} [{p.source}]")
    return 0


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 (one block per command)
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
    swp.add_argument("trace", nargs="?", help="a trace: runs the built-in Stage 3 sweeps on it")
    swp.add_argument(
        "--spec", help="an experiment file (TOML): which parameters to sweep, over what"
    )
    swp.add_argument("--out", help="folder for the results (default: memsim-out/<experiment name>)")
    swp.add_argument("--yes", action="store_true", help="run a grid larger than the limit")
    swp.set_defaults(run=_sweep)

    sen = commands.add_parser("sensitivity", help="vary each assumed parameter alone and rank them")
    sen.add_argument(
        "--spec", required=True, help="a sensitivity file (TOML) with [[vary]] entries"
    )
    sen.add_argument("--out", help="folder for the results (default: memsim-out/<name>)")
    sen.set_defaults(run=_sensitivity)

    kne = commands.add_parser("knee", help="smallest XMEM at which more banks stop helping")
    kne.add_argument("--spec", required=True, help="an experiment file that sweeps num_banks")
    kne.add_argument("--out", help="folder for the results (default: memsim-out/<name>)")
    kne.set_defaults(run=_knee)

    frm = commands.add_parser(
        "formulas", help="check the working-set formulas against saved traces"
    )
    frm.add_argument("--traces", help="folder of traces (default: memsim-out/traces)")
    frm.add_argument(
        "--bank-rows", type=int, nargs="*", default=[1024, 256, 64], help="bank sizes to check at"
    )
    frm.add_argument("--out", help="folder for validation.csv (default: memsim-out/formulas)")
    frm.set_defaults(run=_formulas)

    nst = commands.add_parser(
        "netstats", help="network-only tables: weights, activations, operations, extrapolated knee"
    )
    nst.add_argument(
        "networks", nargs="+", help="network files (TOML) whose layers have an op and a shape"
    )
    nst.add_argument("--bank-rows", type=int, default=1024)
    nst.add_argument(
        "--out", help="folder for netstats.md and netstats.csv (default: memsim-out/netstats)"
    )
    nst.set_defaults(run=_netstats)

    fpg = commands.add_parser("fpga-compare", help="compare the model with FPGA measurements (CSV)")
    fpg.add_argument(
        "--measured", required=True, help="the measurements (CSV; see ipu_memsim/fpga.py)"
    )
    fpg.add_argument(
        "--configs", required=True, help="folder of model configs the 'config' column names"
    )
    fpg.add_argument(
        "--traces", help="folder of traces the 'kernel' column names (default: memsim-out/traces)"
    )
    fpg.add_argument(
        "--fit",
        action="store_true",
        help="fit the bank fill time on half the runs, report the other half",
    )
    fpg.add_argument("--synthetic", type=float, metavar="FILL_CYCLES",
                     help="first WRITE synthetic measurements to --measured, from the model at this fill time")  # fmt: skip
    fpg.add_argument(
        "--noise", type=float, default=0.02, help="relative noise of synthetic measurements"
    )
    fpg.add_argument("--out", help="folder for fpga_report.md (default: memsim-out/fpga)")
    fpg.set_defaults(run=_fpga)

    par = commands.add_parser("params", help="list every tunable parameter (the registry)")
    par.add_argument("model", nargs="?", help="only this model's parameters")
    par.set_defaults(run=_params)

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
