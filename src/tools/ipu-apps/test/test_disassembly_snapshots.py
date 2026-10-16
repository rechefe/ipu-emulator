"""Every corpus program disassembles to the text recorded for it in
``test/baselines/disassembly/`` (see ``test/baselines/disassembly.py``)."""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

import asm_corpus
import disassembly

# Bazel runs the test from the runfiles root of the main repository.
ROOT = Path.cwd()
ENTRIES = asm_corpus.discover(ROOT)


def test_every_corpus_entry_has_exactly_one_snapshot():
    expected = {disassembly.snapshot_name(entry) for entry in ENTRIES}
    recorded = disassembly.recorded(ROOT)
    assert len(expected) == len(ENTRIES), "two corpus entries share a snapshot name"
    missing, extra = sorted(expected - recorded), sorted(recorded - expected)
    assert not missing and not extra, (
        f"Missing snapshots: {missing}\nSnapshots with no corpus entry: {extra}\n"
        f"Record the change with: {disassembly.UPDATE_COMMAND}"
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.key)
def test_disassembly_matches_snapshot(entry):
    name = disassembly.snapshot_name(entry)
    path = ROOT / disassembly.SNAPSHOT_DIR / name
    if not path.is_file():
        pytest.fail(f"No snapshot {name}; run {disassembly.UPDATE_COMMAND}")
    expected = path.read_text()
    actual = disassembly.disassemble(asm_corpus.assemble(ROOT, entry).words)
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                f"{name} (snapshot)",
                f"{name} (current code)",
                n=1,
            )
        )
        pytest.fail(
            f"{entry.key}: disassembly changed. If intended, run "
            f"{disassembly.UPDATE_COMMAND}\n{diff}"
        )
