"""Add CI wall times to the nightly benchmark files (Q71: a 5-10 minute target).

    bazel run //tools/bench:ci_times -- --out DIR --run-id "$GITHUB_RUN_ID"

- ``ci/full_suite_wall_s``: the "Run every non-manual test with coverage" step
  of this nightly run's ``full`` job.
- ``ci/default_tier_wall_s``: the "Run the default test tier" step of the
  ``test`` job of the latest successful ``ci.yml`` run on master.

Reads the GitHub API through ``gh`` (``GH_TOKEN`` and ``GH_REPO`` set). A
measurement that is not available (a failed job, no master run yet) is skipped
with a message; it never fails the job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ipu_bench.result import Result, append_results

FULL = ("full", "Run every non-manual test with coverage", "ci/full_suite_wall_s")
DEFAULT_TIER = ("test", "Run the default test tier", "ci/default_tier_wall_s")


def _seconds(entry: dict) -> float:
    start, end = (
        datetime.fromisoformat(entry[key].replace("Z", "+00:00"))
        for key in ("started_at", "completed_at")
    )
    return (end - start).total_seconds()


def step_wall_s(jobs: list[dict], job_name: str, step_name: str) -> tuple[float, float] | None:
    """(step seconds, job seconds) of a successful step, from the API's job list."""
    for job in jobs:
        if job.get("name") != job_name or job.get("conclusion") != "success":
            continue
        for step in job.get("steps") or ():
            if step.get("name") == step_name and step.get("conclusion") == "success":
                return _seconds(step), _seconds(job)
    return None


def wall_result(jobs: list[dict], spec: tuple[str, str, str], run_ref: str) -> Result | None:
    job_name, step_name, result_name = spec
    found = step_wall_s(jobs, job_name, step_name)
    if found is None:
        print(f"{result_name}: no successful '{job_name}' / '{step_name}' in {run_ref}; skipped")
        return None
    step, job = found
    return Result(result_name, "s", step, extra=f"{run_ref}; whole job {job:.0f} s")


def _gh_json(*args: str) -> Any:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _run_jobs(run_id: str) -> list[dict]:
    return _gh_json("api", f"repos/{{owner}}/{{repo}}/actions/runs/{run_id}/jobs")["jobs"]


def collect(run_id: str) -> list[Result]:
    results = [wall_result(_run_jobs(run_id), FULL, f"nightly run {run_id}")]
    runs = _gh_json(
        "run", "list", "--workflow", "ci.yml", "--branch", "master", "--status", "success",
        "--limit", "1", "--json", "databaseId",
    )  # fmt: skip
    if runs:
        ci_id = str(runs[0]["databaseId"])
        results.append(wall_result(_run_jobs(ci_id), DEFAULT_TIER, f"ci run {ci_id}"))
    else:
        print("ci/default_tier_wall_s: no successful ci.yml run on master; skipped")
    return [r for r in results if r is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the bench output directory")
    parser.add_argument("--run-id", required=True, help="this nightly run's id")
    args = parser.parse_args(argv)
    try:
        results = collect(args.run_id)
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"CI wall times not recorded: {exc} {getattr(exc, 'stderr', '') or ''}")
        return 0
    append_results(results, args.out)
    for r in results:
        print(f"{r.name} = {r.value:.0f} s ({r.extra})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
