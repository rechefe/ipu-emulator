"""Decoded-word field names, emulator field maps and enum descriptors match
their snapshots in ``test/baselines/`` (see ``decode_snapshots.py``)."""

from __future__ import annotations

from pathlib import Path

import baseline_support
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
    baseline_support.fail_if_changed(
        expected, actual, name=name, what=name, update_command=decode_snapshots.UPDATE_COMMAND
    )


def test_word_is_209_bits():
    assert (
        decode_snapshots.word_fields()["total_bits"] == 209
    )  # ADR 0007: half-CR operands, the aaq and aaq_exact slots


def test_renaming_a_token_class_leaves_the_word_fields():
    """Field names come from the ISA's word layout, not from token class names
    (E05-02), so renaming a class changes nothing on the wire."""
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
    assert after == before
