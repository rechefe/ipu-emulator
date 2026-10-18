"""The stage view reads the decoded word slot by slot (E08-07)."""

from __future__ import annotations

from ipu_as.session import AssemblySession
from ipu_debug.pipeline import PIPELINE_STAGES, SLOT_LABELS, pipeline_occupancy, stage_operations
from ipu_isa.codec.word import DecodedWord
from ipu_isa.slots import LAYOUT_ORDER, SLOTS


def _fields(source: str) -> DecodedWord:
    (word,) = AssemblySession(source).assemble().words
    return DecodedWord.from_word(word)


def test_operations_are_in_assembly_order_without_unused_fields():
    operations = dict(stage_operations(_fields("SET LR0 CR1; ADD LR3 LR1 LR2; ACC.ADD;;")))
    assert operations["LR0"] == "SET lr0 cr1"
    assert operations["LR1"] == "ADD lr3 lr1 lr2"
    assert operations["LR2"] == "NOP"
    assert operations["ACC"] == "ACC.ADD"


def test_labels_follow_the_slot_registry():
    expected = [
        label + ("*" if SLOTS[slot].debug_only else "")
        for slot in LAYOUT_ORDER
        for label in SLOTS[slot].instance_labels()
    ]
    assert list(SLOT_LABELS) == expected
    assert [label for label, _ in stage_operations(None)] == expected


def test_every_slot_instance_is_in_one_stage():
    staged = [label for _, labels in PIPELINE_STAGES for label in labels]
    assert sorted(staged) == sorted(SLOT_LABELS)


def test_occupancy_lists_only_busy_slots():
    stages = dict(pipeline_occupancy(_fields("ADD LR3 LR1 LR2; ACC.ADD;;")))
    assert stages["CTRL"] == [("LR0", "ADD lr3 lr1 lr2")]
    assert stages["ACC"] == [("ACC", "ACC.ADD")]
    assert stages["MULT"] == []
