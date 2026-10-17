"""Benchmarks from ``[benchmark]`` (E06-11): the scale kernel timed at three
sizes, held to its reference, through the module and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scale_kernel import write_scale

from ipu_harness import cli
from ipu_harness.benchmark import point_cases, run_benchmark, to_json, to_markdown
from ipu_harness.family import ENTRY_SEPARATOR, load_kernel
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
        (
            '[benchmark]\npoints = [{ rows = 1 }]\nper = ["rows"]\n',
            r"\[benchmark.per\] must be str",
        ),
        ("[benchmark]\npoints = [{ rows = 1 }]\nmax_cycles = 0\n", "at least 1"),
        ("[benchmark]\npoints = [{ rows = 1 }]\nrepeat = 3\n", "unknown key 'repeat'"),
    ],
)
def test_a_wrong_table_names_its_key(tmp_path, table, match):
    with pytest.raises(ManifestError, match=match):
        _with(tmp_path, "\n" + table)


def test_each_point_is_the_default_case_with_its_values(tmp_path):
    manifest = _with(tmp_path)
    cases = point_cases(manifest)
    assert [c.name for c in cases] == [
        "benchmark[rows=1]",
        "benchmark[rows=4,s=3.0]",
        "benchmark[rows=16]",
    ]
    assert cases[0].values == {"rows": 1, "s": 2.0}  # the default case's s
    assert cases[1].values == {"rows": 4, "s": 3.0}
    assert not cases[1].inputs and not cases[1].expected  # a point's shapes differ


def test_every_point_runs_passes_and_is_reported(tmp_path):
    rows = run_benchmark(_with(tmp_path), F.FP32)
    assert [r.result.passed for r in rows] == [True, True, True]
    cycles = [r.result.cycles for r in rows]
    # A fixed setup plus one loop body per row: the gaps grow with the rows.
    body = (cycles[1] - cycles[0]) / 3
    assert body > 0 and cycles[2] - cycles[1] == 12 * body
    assert [r.per_unit for r in rows] == [cycles[0] / 1, cycles[1] / 4, cycles[2] / 16]
    for row in rows:
        used = row.utilization()
        assert used is not None and all(0 <= v <= 1 for v in used.values())
    stats = rows[2].result.stats
    assert stats is not None and rows[2].utilization() == {
        "mult": stats.mult_active_cycles / stats.cycles,
        "acc": stats.acc_active_cycles / stats.cycles,
        "lanes": stats.lane_occupancy,
        "ident": stats.mult_identity_lane_ops / stats.mult_lane_ops,
        "effMAC": stats.effective_mac_utilization,
    }
    table = to_markdown(rows, "rows")
    assert "| point | format | result | cycles | cyc/rows | mult% |" in table
    assert "| rows=16 | fp32 | pass |" in table
    entry = json.loads(to_json(rows))[1]  # the history's format (E01-14)
    assert (entry["name"], entry["unit"], entry["value"]) == (
        "kernel/scale/fp32/rows_4_s_3.0",
        "cycles",
        cycles[1],
    )
    extra = json.loads(entry["extra"])
    assert extra["values"] == {"rows": 4, "s": 3.0} and extra["passed"] is True


def test_the_command_line_runs_the_benchmark(tmp_path, capsys):
    manifest = _with(tmp_path)
    out = tmp_path / "bench.json"
    markdown = tmp_path / "bench.md"
    argv = [str(manifest.directory), "--benchmark", "--format", "all"]
    assert cli.main([*argv, "--json", str(out), "--markdown", str(markdown)]) == 0
    rows = [json.loads(e["extra"]) for e in json.loads(out.read_text())]
    assert {r["format"] for r in rows} == {f.value for f in manifest.formats}
    assert len(rows) == 3 * len(manifest.formats) and all(r["passed"] for r in rows)
    assert markdown.read_text().startswith("# scale benchmark")
    assert "cyc/rows" in capsys.readouterr().out


def test_a_kernel_without_a_benchmark_says_so(tmp_path, capsys):
    plain = write_scale(tmp_path)
    assert cli.main([str(plain.directory), "--benchmark"]) == 1
    assert "has no [benchmark]" in capsys.readouterr().err


def test_a_point_that_does_not_pass_is_shown_and_fails_the_command(tmp_path, capsys):
    manifest = _with(tmp_path, BENCHMARK.replace("max_cycles = 100000", "max_cycles = 5"))
    rows = run_benchmark(manifest, F.FP32)
    assert [r.result.passed for r in rows] == [False, False, False]  # F-9: the cycle limit
    assert "| rows=1 | fp32 | FAIL |" in to_markdown(rows, "rows")
    assert cli.main([str(manifest.directory), "--benchmark"]) == 1
    assert "fault" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--benchmark", "--max-cycles", "9"], "takes no --max-cycles"),
        (["--benchmark", "--case", "one_row"], "takes no --case"),
        (["--json", "out.json"], "--json goes with --benchmark"),
    ],
)
def test_an_option_the_request_would_ignore_is_refused(tmp_path, capsys, extra, match):
    manifest = _with(tmp_path)
    assert cli.main([str(manifest.directory), *extra]) == 1
    assert match in capsys.readouterr().err


def test_a_benchmark_needs_a_reference_to_hold_its_points_to(tmp_path):
    """A point's shapes are its own: no expected file fits it, so without a
    reference it would compare nothing and pass."""
    directory = write_scale(tmp_path, with_reference=False).directory
    toml = directory / "kernel.toml"
    toml.write_text(toml.read_text() + BENCHMARK)
    with pytest.raises(ManifestError, match="needs a \\[golden\\] reference or op"):
        load_manifest(directory)


def test_a_benchmark_needs_a_one_value_default_case(tmp_path):
    directory = write_scale(tmp_path).directory
    toml = directory / "kernel.toml"
    swept = toml.read_text().replace("[cases.default]\nrows = 4", "[cases.default]\nrows = [4, 8]")
    toml.write_text(swept + BENCHMARK)
    with pytest.raises(ManifestError, match="needs a one-value \\[cases.default\\]"):
        load_manifest(directory)


def test_a_point_cannot_move_a_registry_entrys_fixed_parameter(tmp_path):
    template = _with(tmp_path).directory
    registry = template.parent / "registry.toml"
    registry.write_text(
        '[[kernel]]\nname = "scale_r2"\ntemplate = "scale"\nparams = { rows = 2 }\n'
    )
    with pytest.raises(ManifestError, match="rows is fixed at 2 by the registry entry"):
        load_kernel(f"{registry}{ENTRY_SEPARATOR}scale_r2")
