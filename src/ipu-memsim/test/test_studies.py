"""Sweeps and the chart, on a small hand-made trace."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET

import pytest
from support import make_trace

from ipu_memsim.cache_unit import CacheUnit
from ipu_memsim.dram import FixedDram
from ipu_memsim.models import AllMiss
from ipu_memsim.svgplot import Series, line_chart
from ipu_memsim.sweep import run_sweeps, sweep
from ipu_memsim.trace import READ, WRITE


def stream(rows=4096):
    accesses = []
    for i in range(rows):
        accesses += [(4 * i, READ, i), (4 * i, WRITE, 10000 + i)]
    return make_trace(accesses, [("x", "input", 0, rows), ("y", "output", 10000, rows)])


def test_a_sweep_replays_once_per_value_and_skips_impossible_configs():
    trace = stream()
    points = sweep(
        trace, "banks", "cache_unit", [1, 2, 4, 8],
        lambda n: (CacheUnit(int(n), 1024, FixedDram(100, 0.1)), int(n) * 1024 * 128),
    )  # fmt: skip
    # One bank cannot hold two tables: that point is left out.
    assert [p.x for p in points] == [2, 4, 8]
    assert [p.xmem_bytes for p in points] == [262144, 524288, 1048576]
    assert all(p.total_cycles == p.ideal_cycles + p.stall_cycles for p in points)
    # More banks never hurt this streaming kernel.
    totals = [p.total_cycles for p in points]
    assert totals[0] > totals[1] >= totals[2]


def test_a_penalty_sweep_is_linear_for_all_miss():
    points = sweep(stream(), "penalty", "all_miss", [1, 10], lambda p: (AllMiss(int(p)), 0))
    assert [p.stall_cycles for p in points] == [8192, 81920]


def test_run_sweeps_writes_a_table_and_charts(tmp_path):
    written = run_sweeps(stream(), tmp_path)
    assert {p.name for p in written} == {
        "sweeps.csv", "bank_count.svg", "xmem_size.svg", "miss_penalty_toy.svg",
        "dram_bandwidth.svg", "sweep_timing.txt",
    }  # fmt: skip
    with open(tmp_path / "sweeps.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["sweep"] for r in rows} == {
        "bank_count",
        "xmem_size",
        "miss_penalty_toy",
        "dram_bandwidth",
    }
    sizes = {float(r["x"]) for r in rows if r["sweep"] == "xmem_size"}
    assert sizes == {0.5, 1, 2, 4, 8}  # below and above today's 2 MB
    for row in rows:
        assert int(row["total_cycles"]) == int(row["ideal_cycles"]) + int(row["stall_cycles"])
    for name in ("bank_count.svg", "xmem_size.svg"):
        text = (tmp_path / name).read_text()
        ET.fromstring(text)  # well-formed
        assert "TOY MODEL" in text
    # The same trace gives the same table.
    first = (tmp_path / "sweeps.csv").read_bytes()
    run_sweeps(stream(), tmp_path / "again")
    assert (tmp_path / "again" / "sweeps.csv").read_bytes() == first


def test_the_chart_names_every_series_and_marks_every_point():
    svg = line_chart(
        [Series("a & b", [(1, 10), (2, 20)]), Series("c", [(1, 5), (2, 7)])],
        title="t", subtitle="s", x_label="x", y_label="y",
    )  # fmt: skip
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}polyline")) == 2
    assert "a &amp; b" in svg  # names are escaped
    # 2 points + 1 legend marker per series; the two series use different shapes.
    assert len(root.findall(f"{ns}circle")) == 3 and len(root.findall(f"{ns}rect")) == 1 + 3
    with pytest.raises(ValueError, match="1 to 8 series"):
        line_chart([], title="t", subtitle="s", x_label="x", y_label="y")


# -- the network table ---------------------------------------------------------

NETWORK = """
name = "tiny"

[[layer]]
type = "elementwise"
name = "residual 64"
kernel = "residual_add"
count = 3

[[layer]]
type = "elementwise"
name = "residual 96"
kernel = "residual_add"
values = { num_channels = 96 }

[[layer]]
type = "mlp"
name = "no such kernel"
kernel = "does_not_exist"

[[missing]]
type = "global_average_pool"
note = "no kernel"
"""

FAST = '[model]\nkind = "cache_unit"\nnum_banks = 16\n[dram]\nkind = "fixed"\nlatency_cycles = 10\nrows_per_cycle = 1.0\n'
SMALL = '[model]\nkind = "cache_unit"\nnum_banks = 2\n'


def test_the_network_table(tmp_path):
    from ipu_memsim.network import run_network

    (tmp_path / "net.toml").write_text(NETWORK)
    (tmp_path / "fast.toml").write_text(FAST)
    (tmp_path / "small.toml").write_text(SMALL)
    written = run_network(
        tmp_path / "net.toml", tmp_path / "out", [tmp_path / "fast.toml", tmp_path / "small.toml"]
    )
    assert [p.name for p in written] == ["layers.csv", "layer_types.csv", "layer_types.md"]
    with open(tmp_path / "out" / "layers.csv", newline="") as handle:
        layers = list(csv.DictReader(handle))
    first, second, broken = layers
    # residual_add: 5 cycles per channel + 5; a, b and y are 64 rows each.
    assert (first["ideal_cycles"], second["ideal_cycles"]) == ("325", "485")
    assert (first["footprint_rows"], first["input_rows"], first["output_rows"]) == (
        "192",
        "128",
        "64",
    )
    # Two demand fetches of 64 rows, one after the other: 2 * (10 + 64).
    assert first["stall_fast"] == "148"
    # Two banks cannot hold three tables: the layer says so instead of vanishing.
    assert first["stall_small"] == "-1" and "3 tables need at least 3 banks" in first["note"]
    assert broken["ideal_cycles"] == "0" and "unknown kernel" in broken["note"]

    with open(tmp_path / "out" / "layer_types.csv", newline="") as handle:
        types = {row["layer type"]: row for row in csv.DictReader(handle)}
    # Weighted by count: 3 * 325 + 485 cycles, 3 * 148 + (2 * (10 + 96)) stall cycles.
    assert types["elementwise"]["ideal cycles"] == str(3 * 325 + 485)
    assert types["elementwise"]["stall: fast"].startswith(str(3 * 148 + 212))
    assert types["mlp"]["not measured"] == "no such kernel"
    assert types["global_average_pool"]["not measured"] == "NO KERNEL: no kernel"
    text = (tmp_path / "out" / "layer_types.md").read_text()
    assert "TOY-MODEL" in text and "| global_average_pool |" in text
    # The traces were kept: a second run reuses them and gives the same table.
    assert len(list((tmp_path / "out" / "traces").iterdir())) == 2
    before = (tmp_path / "out" / "layers.csv").read_bytes()
    run_network(
        tmp_path / "net.toml", tmp_path / "out", [tmp_path / "fast.toml", tmp_path / "small.toml"]
    )
    assert (tmp_path / "out" / "layers.csv").read_bytes() == before


# -- the DRAMsim3 adapter, against a stand-in for the simulator ------------------

# A stand-in for dramsim3main.out: it completes one transaction per cycle after
# a 20-cycle start, and writes the statistics file DRAMsim3 writes.
FAKE_DRAMSIM3 = """#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
cycles = int(args[args.index("-c") + 1])
lines = open(args[args.index("-t") + 1]).read().split("\\n")[:-1]
done = max(0, min(len(lines), cycles - 20))
writes = "WRITE" in lines[0]
stats = {"0": {"num_reads_done": 0 if writes else done, "num_writes_done": done if writes else 0}}
json.dump(stats, open(args[args.index("-o") + 1] + "/dramsim3.json", "w"))
sys.exit(133)  # the real program aborts on exit on macOS; the adapter must cope
"""
INI = "protocol = HBM\nbus_width = 128\nBL = 4\ntCK = 2\n"


def test_the_dramsim3_adapter(tmp_path):
    from ipu_memsim import dram_trace
    from ipu_memsim.cache_unit import DmaTransfer
    from ipu_memsim.dram import make_dram

    binary = tmp_path / "fake_dramsim3"
    binary.write_text(FAKE_DRAMSIM3)
    binary.chmod(0o755)
    ini = tmp_path / "hbm.ini"
    ini.write_text(INI)
    assert dram_trace.burst_bytes(ini) == 64 and dram_trace.clock_ns(ini) == 2.0

    # 16 rows of 128 bytes are 32 transactions of 64 bytes.
    assert dram_trace.write_fill_trace(tmp_path / "t", 16, 64) == 32
    assert (tmp_path / "t").read_text().splitlines()[:2] == ["0x0 READ 0", "0x40 READ 0"]
    # The stand-in finishes them at DRAM cycle 20 + 32.
    assert dram_trace.dram_cycles_to_read(binary, ini, 16) == 52
    # In IPU cycles of 1 ns that is twice as many (tCK = 2 ns).
    table = dram_trace.measure_fill_table(binary, ini, sizes=(16, 64), ipu_clock_ns=1.0)
    assert table == {"fill": {"16": 104, "64": 296}, "drain": {"16": 104, "64": 296}}
    backend = make_dram({"kind": "table", **table})
    assert backend.transfer_cycles(64, write=False) == 296

    # A replay's transfers as one trace: tables 16 MiB apart, pages inside.
    transfers = [DmaTransfer(500, 600, "y", 0, 1, True), DmaTransfer(0, 100, "x", 1, 1, False)]
    dram_trace.write_transfer_trace(tmp_path / "all", transfers, 64, cycle_scale=0.5)
    assert (tmp_path / "all").read_text().splitlines() == [
        "0x20000 READ 0", "0x20040 READ 0", "0x1000000 WRITE 250", "0x1000040 WRITE 250",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="no 'bus_width = ' line"):
        dram_trace.burst_bytes(tmp_path / "all")
