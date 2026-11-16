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
    with pytest.raises(ValueError, match="1 to 4 series"):
        line_chart([], title="t", subtitle="s", x_label="x", y_label="y")
