"""Decoded-word field names, emulator field maps and enum descriptors match
their snapshots in ``test/baselines/`` (see ``decode_snapshots.py``)."""

from __future__ import annotations

import difflib
from pathlib import Path

import decode_snapshots
import pytest

from ipu_as import reg
from ipu_as.compound_inst import CompoundInst

# Bazel runs the test from the runfiles root of the main repository.
ROOT = Path.cwd()


@pytest.mark.parametrize("name", sorted(decode_snapshots.SNAPSHOTS))
def test_snapshot_matches(name):
    expected = (ROOT / decode_snapshots.SNAPSHOT_DIR / name).read_text()
    actual = decode_snapshots.render(decode_snapshots.SNAPSHOTS[name]())
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                f"{name} (snapshot)",
                f"{name} (current code)",
            )
        )
        pytest.fail(f"{name} changed. If intended, run {decode_snapshots.UPDATE_COMMAND}\n{diff}")


def test_word_is_186_bits():
    assert decode_snapshots.word_fields()["total_bits"] == 186


def test_renaming_a_token_class_changes_word_fields():
    """Field names follow token class names today, so the snapshot catches a
    class rename even when the encoding is unchanged."""
    before = decode_snapshots.word_fields()
    # monkeypatch cannot restore __name__ (it is not in the class __dict__).
    # get_fields() is computed once (E05-01), so recompute it around the rename.
    original = reg.LrRegField.__name__
    reg.LrRegField.__name__ = "RenamedLrRegField"
    CompoundInst.get_fields.cache_clear()
    try:
        after = decode_snapshots.word_fields()
    finally:
        reg.LrRegField.__name__ = original
        CompoundInst.get_fields.cache_clear()
    assert after != before
    assert any("renamed_lr_reg_field" in name for name, _ in after["fields_lsb_first"])
