"""One ISA entry is enough for the ISA package (E03-24, Part A).

With ``TESTNEG`` added to the LR slot and nothing else changed, every artifact
``ipu_isa`` derives follows: opcode, layout, slot codec, disassembly, syntax,
timing check and the F-1 reserved encodings. The generators' half is
``ipu-gen/test/test_one_edit_generators.py``; the end-to-end half (assembler
and emulator, in a subprocess) is ``ipu-emu/test/test_one_edit_end_to_end.py``.
"""

from __future__ import annotations

import pytest
from one_edit import MNEMONIC, view_with_dummy

from ipu_isa.codec.disasm import operation_text
from ipu_isa.codec.slot import codecs
from ipu_isa.faults import reserved_encodings
from ipu_isa.slots import Slot
from ipu_isa.view import isa


@pytest.fixture(scope="module")
def view():
    return view_with_dummy()


def test_it_takes_the_next_free_opcode_and_moves_nothing(view):
    before = {(s, i.mnemonic): i.opcode for s in Slot for i in isa().instructions(s)}
    after = {(s, i.mnemonic): i.opcode for s in Slot for i in view.instructions(s)}
    assert after.pop((Slot.LR, MNEMONIC)) == len(isa().instructions(Slot.LR))
    assert after == before


def test_no_width_changes(view):
    for slot in Slot:
        assert view.layout.slot_width(slot) == isa().layout.slot_width(slot), slot
    assert view.layout.word_bits() == isa().layout.word_bits() == 186


def test_the_slot_codec_round_trips_and_disassembles(view):
    lr = codecs(view).slots[Slot.LR]
    value = lr.encode(MNEMONIC, [1, 0])
    decoded = lr.decode(value)
    assert decoded.instruction.mnemonic == MNEMONIC
    assert decoded.operands == (1, 0)
    assert decoded.unused_bits == 0
    assert operation_text(decoded, isa=view) == "TESTNEG lr1 lr0"


def test_the_syntax_is_derived(view):
    assert view.instruction(Slot.LR, MNEMONIC).syntax == "TESTNEG dest src"


def test_the_timing_check_accepts_it(view):
    # load_isa ran the timing validator; the declared write and read are kept.
    instruction = view.instruction(Slot.LR, MNEMONIC)
    assert [(op.name, op.write, op.read and op.read.value) for op in instruction.operands] == [
        ("dest", True, None),
        ("src", False, "cycle_start"),
    ]


def test_it_takes_one_reserved_lr_opcode(view):
    before = reserved_encodings(isa())[Slot.LR].opcodes
    assert before[0] == view.instruction(Slot.LR, MNEMONIC).opcode
    assert reserved_encodings(view)[Slot.LR].opcodes == before[1:]
