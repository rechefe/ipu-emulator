"""Benchmarks from ``[benchmark]`` (E06-11): the scale kernel timed at three
sizes, held to its reference, through the module and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scale_kernel import write_scale

from ipu_harness import cli
from ipu_harness.benchmark import points, render, run_benchmark, to_json
from ipu_harness.manifest import ManifestError, load_manifest
from ipu_isa.formats import ElementFormat as F

BENCHMARK = """
[benchmark]
points = [{ rows = 1 }, { rows = 4, s = 3.0 }, { rows = 16 }]
per = "rows"
max_cycles = 100000
"""


def _with(tmp_path: Path, table: str = BENCHMARK):
    directory = write_scale(tmp_path).directory
    toml = directory / "kernel.toml"
    toml.write_text(toml.read_text() + table)
    return load_manifest(directory)


def test_the_table_loads(tmp_path):
    bench = _with(tmp_path).benchmark
    assert bench is not None
    assert [dict(p) for p in bench.points] == [{"rows": 1}, {"rows": 4, "s": 3.0}, {"rows": 16}]
    assert (bench.per, bench.max_cycles) == ("rows", 100000)
    assert write_scale(tmp_path / "plain").benchmark is None


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ("[benchmark]\npoints = []\n", "needs a non-empty points list"),
        ("[benchmark]\npoints = [{ rows = [1, 2] }]\n", "is one value, not a list"),
        ("[benchmark]\npoints = [{ rowz = 1 }]\n", "unknown key 'rowz'"),
        ("[benchmark]\npoints = [{ rows = 0 }]\n", "outside its domain"),
        ('[benchmark]\npoints = [{ rows = 1 }]\nper = "s"\n', "per 's' is no int parameter"),
        ("[benchmark]\npoints = [{ rows = 1 }]\nmax_cycles = 0\n", "at least 1"),
        ("[benchmark]\npoints = [{ rows = 1 }]\nrepeat = 3\n", "unknown key 'repeat'"),
    ],
)
def test_a_wrong_table_names_its_key(tmp_path, table, match):
    with pytest.raises(ManifestError, match=match):
        _with(tmp_path, "\n" + table)


def test_each_point_is_the_default_case_with_its_values(tmp_path):
    manifest = _with(tmp_path)
    cases = points(manifest)
    assert [c.name for c in cases] == [
        "benchmark[rows=1]",
        "benchmark[rows=4,s=3.0]",
        "benchmark[rows=16]",
    ]
    assert cases[0].values["s"] == cases[2].values["s"] != 3.0  # the default case's s
    assert not cases[1].inputs and not cases[1].expected  # a point's shapes differ


def test_every_point_runs_passes_and_is_reported(tmp_path):
    rows = run_benchmark(_with(tmp_path), F.FP32)
    assert [r.result.passed for r in rows] == [True, True, True]
    cycles = [r.result.cycles for r in rows]
    assert cycles[0] < cycles[1] < cycles[2]  # one loop iteration per row
    assert [r.per_unit for r in rows] == [cycles[0] / 1, cycles[1] / 4, cycles[2] / 16]
    for row in rows:
        used = row.utilization()
        assert used is not None and all(0 <= v <= 1 for v in used.values())
        assert used["effMAC"] == row.result.mac_utilization
    table = render(rows, "rows")
    assert "| point | format | result | cycles | cyc/rows | mult% |" in table
    assert "| rows=16 | fp32 | pass |" in table
    assert json.loads(to_json(rows))[2]["cycles"] == cycles[2]


def test_the_command_line_runs_the_benchmark(tmp_path, capsys):
    manifest = _with(tmp_path)
    out = tmp_path / "bench.json"
    markdown = tmp_path / "bench.md"
    argv = [str(manifest.directory), "--benchmark", "--format", "all"]
    assert cli.main([*argv, "--json", str(out), "--markdown", str(markdown)]) == 0
    rows = json.loads(out.read_text())
    assert {r["format"] for r in rows} == {f.value for f in manifest.formats}
    assert len(rows) == 3 * len(manifest.formats) and all(r["passed"] for r in rows)
    assert markdown.read_text().startswith("# scale benchmark")
    assert "cyc/rows" in capsys.readouterr().out


def test_a_kernel_without_a_benchmark_says_so(tmp_path, capsys):
    plain = write_scale(tmp_path)
    assert cli.main([str(plain.directory), "--benchmark"]) == 1
    assert "has no [benchmark]" in capsys.readouterr().err
