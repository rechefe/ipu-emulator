"""VLIW instruction decoder and legacy execution API.

This module provides:
- Instruction word decoding (decode_instruction_word)
- Binary instruction file loading (load_binary_instructions)

The actual instruction execution is handled by the handlers of ipu_emu.slots
(one module per slot), which ipu_emu.ipu binds from instruction_spec metadata.

Instruction format
~~~~~~~~~~~~~~~~~~
An instruction is stored as a ``dict[str, int]`` whose keys match the
wire field names of ``ipu_isa.codec.word.WORD_LAYOUT``, e.g.::

    {
        "break_inst_token_0_break_inst_opcode": 2,   # NOP (break slot)
        "load_inst_token_0_load_inst_opcode": 3,     # NOP (load slot)
        "store_inst_token_0_store_inst_opcode": 1,   # NOP (store slot)
        "acc_store_inst_token_0_acc_store_inst_opcode": 1,   # NOP (acc_store slot)
        ...
    }

These dicts (``DecodedWord``) are produced by :func:`decode_instruction_word`.
"""

from __future__ import annotations

# Re-exported: emulator.py imports Ipu, and tests import both, from here.
from ipu_emu.ipu import BreakResult, Ipu  # noqa: F401  (re-exported)
from ipu_isa.codec.word import DecodedWord, opcode_field, unpack_program
from ipu_isa.slots import SLOTS
from ipu_isa.view import isa

# ---------------------------------------------------------------------------
# Instruction words and program files: the ISA's word codec (E03-16)
# ---------------------------------------------------------------------------

#: Decode a raw instruction word into a field dict.
decode_instruction_word = DecodedWord.from_word

#: Decode a binary program file (``ipu-as assemble --format bin``).
load_binary_instructions = unpack_program


def emulator_only_instructions(instructions) -> set[str]:
    """Names of the emulator-only instructions a decoded program contains.

    An instruction is emulator-only when it sits in a debug-only slot
    (``SlotInfo.debug_only``, today only ``acc_store``) and is not that slot's
    ``NOP``. Reads opcodes from the
    decoded words of :func:`load_binary_instructions`, so it does not depend
    on mnemonic spelling.
    """
    infos = [info for info in SLOTS.values() if info.debug_only]
    for info in infos:
        if info.count != 1:
            raise NotImplementedError(
                f"emulator-only slot {info.slot.value!r} appears {info.count} times"
            )
    fields = [(info.slot, opcode_field(info.slot)) for info in infos]
    found: set[str] = set()
    for inst in instructions:
        for slot, field in fields:
            name = isa().by_opcode(slot, inst[field]).mnemonic
            if name != "NOP":
                found.add(name)
    return found
