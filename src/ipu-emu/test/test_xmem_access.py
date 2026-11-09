"""Every XMEM instruction names its base CR and offset LR to the access observer."""

from __future__ import annotations

from ipu_emu.xmem_access import address_operands
from ipu_isa.slots import Slot
from ipu_isa.view import isa

#: Each instruction that reads or writes an XMEM row, and its (base, offset)
#: operands, from the instruction reference (memory-access.md), by hand.
XMEM_INSTRUCTIONS = {
    "LD.MULT": ("base", "offset"),
    "LD.MULT.CYCLIC": ("base", "offset"),
    "LD.MASK": ("base", "offset"),
    "LD.LUT": ("table_id", "table_offset"),
    "ACTIVATE.QUANTIZE": ("table_id", "table_offset"),
    "ACTIVATE.EXACT": ("table_id", "table_offset"),
}

#: The slots whose instructions touch XMEM.
XMEM_SLOTS = (Slot.LOAD, Slot.AAQ, Slot.AAQ_EXACT)


def test_every_xmem_instruction_has_its_address_pair():
    for mnemonic, pair in XMEM_INSTRUCTIONS.items():
        assert address_operands(mnemonic) == pair, mnemonic


def test_no_other_instruction_of_an_xmem_slot_goes_unlisted():
    # A new XMEM instruction fails here until it is added above (and its
    # operands named as ipu_emu.xmem_access.ADDRESS_OPERANDS expects).
    for slot in XMEM_SLOTS:
        for instruction in isa().instructions(slot):
            if instruction.mnemonic != "NOP":
                assert instruction.mnemonic in XMEM_INSTRUCTIONS, instruction.mnemonic


def test_an_instruction_without_a_row_address_has_no_pair():
    assert address_operands("MULT.VE") is None
    assert address_operands("NOP") is None
