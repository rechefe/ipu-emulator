"""The fault list and the arithmetic events as data; F-1 derived (E03-15)."""

from __future__ import annotations

import copy

from ipu_isa.faults import (
    FAULTS,
    ArithmeticEvent,
    Detector,
    FaultId,
    reserved_encodings,
    reserved_operand_codes,
)
from ipu_isa.instruction_spec import INSTRUCTION_SPEC, PSEUDO_INSTRUCTION_SPEC
from ipu_isa.operand_types import OPERAND_TYPES
from ipu_isa.slots import SLOTS, Slot
from ipu_isa.tags import Tag
from ipu_isa.view import isa, load_isa


def test_nine_faults_in_priority_order():
    assert list(FAULTS) == list(FaultId)
    assert [int(fault_id) for fault_id in FAULTS] == list(range(1, 10))
    for fault_id, info in FAULTS.items():
        assert info.id is fault_id
        assert fault_id.label == f"F-{int(fault_id)}"
    # IPU-FLT-8: the lowest number wins, so priority is the number itself.
    assert min(FaultId.F5, FaultId.F3) is FaultId.F3


def test_only_f8_is_not_modelled():
    assert [info.id for info in FAULTS.values() if not info.modelled] == [FaultId.F8]


def test_every_fault_is_enabled_at_reset():
    assert all(info.enabled_at_reset for info in FAULTS.values())


def test_detectors():
    cache_agent = {fid for fid, info in FAULTS.items() if info.detected_in is Detector.CACHE_AGENT}
    assert cache_agent == {FaultId.F6, FaultId.F7, FaultId.F8}


def test_arithmetic_events_are_not_faults():
    assert [str(event) for event in ArithmeticEvent] == [
        "acc_overflow",
        "nan_result",
        "quant_saturation",
    ]


def test_reserved_opcodes_of_todays_isa():
    reserved = reserved_encodings(isa())
    assert set(reserved) == set(isa().layout.slots_with_tag(Tag.HARDWARE))
    counts = {slot: len(entry.opcodes) for slot, entry in reserved.items()}
    assert counts == {
        Slot.COND: 1,
        Slot.LR: 7,
        Slot.LOAD: 0,
        Slot.MULT: 2,
        Slot.ACC: 3,
        Slot.AAQ: 0,
        Slot.STORE: 0,
    }
    assert reserved[Slot.COND].opcodes == (7,)
    assert reserved[Slot.LR].opcode_bits == 4
    assert reserved[Slot.STORE].empty


def _operand_codes(slot):
    return {r.operand_type: r.codes for r in reserved_encodings(isa())[slot].operands}


def test_reserved_operand_codes_of_todays_isa():
    assert _operand_codes(Slot.LOAD) == {"MultStageReg": (2, 3)}
    assert _operand_codes(Slot.MULT) == {"MultStageReg": (2, 3)}
    assert _operand_codes(Slot.ACC) == {
        "ElementsInRow": (3,),
        "HorizontalStride": (3,),
        "VerticalStride": (3,),
    }
    assert _operand_codes(Slot.AAQ) == {"ActivationFn": (13, 14, 15)}
    # INCR_MOD_POW2 stores k - 1: k in 1..9 uses codes 0..8.
    assert _operand_codes(Slot.LR) == {"LrModPow2KImmediate": tuple(range(9, 16))}
    assert _operand_codes(Slot.COND) == {}


def test_full_width_operands_reserve_nothing():
    for name in ("LrIdx", "LcrIdx", "AddbiImmediate", "BreakImmediate", "Label"):
        bits = isa().layout.operand_bits[name]
        assert reserved_operand_codes(OPERAND_TYPES[name], bits) == (), name


def test_emulator_slots_have_no_entry():
    reserved = reserved_encodings(isa())
    assert Slot.BREAK not in reserved
    assert Slot.ACC_STORE not in reserved


def test_a_new_instruction_takes_a_reserved_opcode():
    spec = copy.deepcopy(INSTRUCTION_SPEC)
    spec["cond"]["NEW_OP"] = copy.deepcopy(spec["cond"]["NOP"])
    changed = load_isa(spec, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)
    assert reserved_encodings(changed)[Slot.COND].opcodes == ()
