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
    # Every opcode past the slot's last instruction is reserved, so the counts
    # follow the spec; the field widths are wire format and stay pinned.
    for slot, entry in reserved.items():
        used = len(INSTRUCTION_SPEC[slot.value])
        assert entry.opcodes == tuple(range(used, 1 << entry.opcode_bits)), slot
    assert {slot: entry.opcode_bits for slot, entry in reserved.items()} == {
        Slot.COND: 3,
        Slot.LR: 4,
        Slot.LOAD: 2,
        Slot.MULT: 3,
        Slot.ACC: 4,
        Slot.AAQ: 1,
        Slot.STORE: 1,
    }
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
    # INC.MODP2 stores k - 1: k in 1..9 uses codes 0..8.
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
    spec["cond"]["NEWOP"] = copy.deepcopy(spec["cond"]["NOP"])
    changed = load_isa(spec, PSEUDO_INSTRUCTION_SPEC, SLOTS, OPERAND_TYPES)
    before = reserved_encodings(isa())[Slot.COND].opcodes
    assert reserved_encodings(changed)[Slot.COND].opcodes == before[1:]


# By hand, from each fault's summary in docs/content/requirements/faults.md.
EXPECTED_INSTRUCTION_FAULTS = {
    "LD.MULT": ("F-1", "F-6", "F-7"),  # MultStageReg: code 2..3 reserved
    "LD.MULT.CYCLIC": ("F-4", "F-6", "F-7"),
    "LD.MASK": ("F-6", "F-7"),
    "ST.AAQ": ("F-6", "F-7"),
    "MULT.RC.VV": ("F-1",),  # ra is a MultStageReg
    "ACC.STRIDE": ("F-1",),  # HorizontalStride/VerticalStride reserved3, ElementsInRow code 3
    "ACC.RESHAPE": ("F-5",),
    "ACTIVATE.QUANTIZE": ("F-1",),  # ActivationFn: codes 13..15 reserved
    "SET": ("F-3",),
    "ADD": ("F-3",),
    "SUB": ("F-3",),
    "INC.MODP2": ("F-1", "F-3"),  # k: codes 9..15 reserved
    "INC": ("F-3",),
    "DEC": ("F-3",),
    "ADDB": ("F-3",),
    "ADDBI": ("F-3",),
    "BEQ": ("F-2",),
    "BNE": ("F-2",),
    "BLT.S": ("F-2",),
    "BGE.S": ("F-2",),
    "BR": ("F-2",),
}


def test_instruction_faults():
    from ipu_isa.faults import instruction_faults

    actual = {
        i.mnemonic: tuple(f.label for f in instruction_faults(isa(), i))
        for slot in Slot
        for i in isa().instructions(slot)
    }
    raising = {mnemonic: faults for mnemonic, faults in actual.items() if faults}
    assert raising == EXPECTED_INSTRUCTION_FAULTS
    # HALT, NOP, the MULT/ACC instructions without reserved codes, and the
    # emulator-only ST.ACC, BREAK and BREAK.IFEQ raise none.
    for mnemonic in ("HALT", "NOP", "ACC.ADD", "MULT.VE", "ST.ACC", "BREAK", "BREAK.IFEQ"):
        assert actual[mnemonic] == ()
