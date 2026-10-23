"""Every generated artifact other than the SV package matches its snapshot in
``test/baselines/generated/`` (see ``generated_artifacts.py``).

Normalizations applied before comparing are listed in
``generated_artifacts.NORMALIZATIONS`` (none today)."""

from __future__ import annotations

from pathlib import Path

import baseline_support
import generated_artifacts
import pytest

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
    _, actual = ARTIFACTS[name]
    expected = (SNAPSHOTS / name).read_text(encoding="utf-8")
    baseline_support.fail_if_changed(
        expected,
        actual,
        name=name,
        what=name,
        update_command=generated_artifacts.UPDATE_COMMAND,
        max_lines=MAX_DIFF_LINES,
    )
