"""Smoke test for //tools/bench: every module runs, and the files are valid."""

from __future__ import annotations

import json

import pytest

from ipu_bench import bench, ci_times
from ipu_bench.result import BIGGER_FILE, SMALLER_FILE, TINY_ENV, Result, append_results


@pytest.fixture(autouse=True)
def _tiny(monkeypatch):
    monkeypatch.setenv(TINY_ENV, "1")


def test_every_module_runs_and_writes_valid_files(tmp_path):
    assert {"imports", "assemble", "kernels"} <= bench.benchmark_modules().keys()
    assert bench.main(["--quick", "--out", str(tmp_path)]) == 0
    smaller = json.loads((tmp_path / SMALLER_FILE).read_text())
    bigger = json.loads((tmp_path / BIGGER_FILE).read_text())
    names = {e["name"] for e in smaller + bigger}
    for name in (
        "import/ipu_as",
        "import/ipu_emu",
        "import/ipu_apps",
        "assemble/all_kernels",
        "kernel/identity/peak_rss_mb",
    ):
        assert name in {e["name"] for e in smaller}, name
    for name in ("assemble/words_per_s", "kernel/identity/cycles_per_s"):
        assert name in {e["name"] for e in bigger}, name
    assert len(names) == len(smaller) + len(bigger)
    for entry in smaller + bigger:
        assert set(entry) <= {"name", "unit", "value", "extra"}
        assert entry["value"] > 0, entry


def test_result_rejects_bad_entries():
    with pytest.raises(ValueError):
        Result("nogroup", "s", 1.0)
    with pytest.raises(ValueError):
        Result("a/b", "s", float("nan"))
    with pytest.raises(ValueError):
        Result("a/b", "", 1.0)


def test_unknown_module_is_refused():
    with pytest.raises(SystemExit):
        bench.run_all(quick=True, only=["no_such_module"])


def _job(name, step, start, end, conclusion="success"):
    return {
        "name": name,
        "conclusion": conclusion,
        "started_at": "2026-10-16T02:00:00Z",
        "completed_at": "2026-10-16T03:00:00Z",
        "steps": [
            {"name": step, "conclusion": conclusion, "started_at": start, "completed_at": end}
        ],
    }


def test_ci_wall_time_from_job_list():
    jobs = [
        _job("stacks", "other", "2026-10-16T02:00:00Z", "2026-10-16T02:01:00Z"),
        _job(*ci_times.FULL[:2], "2026-10-16T02:10:00Z", "2026-10-16T02:40:30Z"),
    ]
    result = ci_times.wall_result(jobs, ci_times.FULL, "nightly run 1")
    assert result.name == "ci/full_suite_wall_s"
    assert result.value == 1830
    assert "whole job 3600 s" in result.extra


def test_ci_wall_time_skips_failed_jobs():
    jobs = [_job(*ci_times.DEFAULT_TIER[:2], "2026-10-16T02:10:00Z", "2026-10-16T02:15:00Z")]
    jobs[0]["conclusion"] = "failure"
    assert ci_times.wall_result(jobs, ci_times.DEFAULT_TIER, "ci run 2") is None


def test_append_keeps_existing_results(tmp_path):
    append_results(
        [Result("a/x", "s", 1.0), Result("a/y", "1/s", 2.0, bigger_is_better=True)], tmp_path
    )
    append_results([Result("ci/z", "s", 3.0)], tmp_path)
    smaller = json.loads((tmp_path / SMALLER_FILE).read_text())
    bigger = json.loads((tmp_path / BIGGER_FILE).read_text())
    assert [e["name"] for e in smaller] == ["a/x", "ci/z"]
    assert [e["name"] for e in bigger] == ["a/y"]
