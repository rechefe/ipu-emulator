"""Every generated artifact other than the SV package matches its snapshot in
``test/baselines/generated/`` (see ``generated_artifacts.py``)."""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

import generated_artifacts

# Bazel runs the test from the runfiles root of the main repository.
SNAPSHOTS = Path.cwd() / generated_artifacts.SNAPSHOT_DIR
ARTIFACTS = generated_artifacts.generate()
MAX_DIFF_LINES = 80


def test_every_snapshot_is_generated():
    recorded = {p.relative_to(SNAPSHOTS).as_posix() for p in SNAPSHOTS.rglob("*") if p.is_file()}
    assert recorded == set(ARTIFACTS), (
        f"Snapshots on disk {sorted(recorded)} differ from generated {sorted(ARTIFACTS)}; "
        f"run {generated_artifacts.UPDATE_COMMAND}"
    )


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_artifact_matches_snapshot(name):
    function, actual = ARTIFACTS[name]
    expected = (SNAPSHOTS / name).read_text(encoding="utf-8")
    if actual == expected:
        return
    diff = list(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            f"{name} (snapshot)",
            f"{name} (from {function})",
        )
    )
    shown = "".join(diff[:MAX_DIFF_LINES])
    if len(diff) > MAX_DIFF_LINES:
        shown += f"... {len(diff) - MAX_DIFF_LINES} more diff lines\n"
    pytest.fail(
        f"{name} changed. If intended, run {generated_artifacts.UPDATE_COMMAND}\n{shown}"
    )
