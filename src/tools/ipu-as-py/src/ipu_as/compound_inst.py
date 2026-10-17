"""The instruction word as the assembler's callers have always used it.

A thin shim over ``ipu_as.encode`` and the ``ipu_isa`` codec (E05-02): the
encoding, the field layout and the disassembly text (``ipu_isa.codec.disasm``)
all come from there. The
debugger, the emulator tests and the generators still call these names; E08
moves the generators onto the codec directly.
"""

import functools

from ipu_as.encode import WILDCARD_OPCODES, encode_word
from ipu_isa.codec.disasm import disassemble
from ipu_isa.codec.word import WORD_LAYOUT
from ipu_isa.view import isa

__all__ = ["INST_SEPARATOR", "WILDCARD_OPCODES", "CompoundInst"]

INST_SEPARATOR = ";"


class CompoundInst:
    """One parsed compound instruction, validated and encoded on construction."""

    def __init__(self, instructions: dict):
        """Place and encode ``instructions`` (one entry of ``ASTBuilder``'s output).

        Raises:
            SlotError: an unknown mnemonic or a full slot.
            OperandError: a wrong operand count or an invalid operand.
            LabelError: an undefined label.
        """
        self._word = encode_word(instructions)

    def encode(self) -> int:
        """The instruction word."""
        return self._word

    @classmethod
    def decode(cls, value: int) -> str:
        """The assembly text of the word ``value`` (``ipu_isa.codec.disasm.disassemble``)."""
        return disassemble(value)

    @classmethod
    @functools.cache
    def bits(cls) -> int:
        """The instruction word width."""
        return isa().layout.word_bits()

    @classmethod
    @functools.cache
    def get_fields(cls) -> tuple[tuple[str, int], ...]:
        """``(name, bits)`` of every instruction word field, LSB first (``WORD_LAYOUT``).

        Computed once; ``CompoundInst.get_fields.cache_clear()`` recomputes it.
        """
        return tuple((field.name, field.bits) for field in WORD_LAYOUT)
