"""The assembler's per-slot instruction classes cover the slot registry (E03-06)."""

from __future__ import annotations

import pytest

from ipu_as.compound_inst import SLOT_TO_INST, check_slot_to_inst
from ipu_isa.schema import SpecError
from ipu_isa.slots import Slot


def test_the_real_table_covers_every_slot():
    check_slot_to_inst(SLOT_TO_INST)


def test_a_missing_slot_is_named():
    partial = {slot: cls for slot, cls in SLOT_TO_INST.items() if slot is not Slot.AAQ}
    # compound_inst runs this same check on its own table at import.
    with pytest.raises(SpecError, match=r"missing \['aaq'\]"):
        check_slot_to_inst(partial)


def test_a_class_of_another_slot_is_rejected():
    swapped = dict(SLOT_TO_INST, **{Slot.AAQ: SLOT_TO_INST[Slot.ACC]})
    with pytest.raises(SpecError, match=r"SLOT_TO_INST\[aaq\] is AccInst, a acc class"):
        check_slot_to_inst(swapped)
