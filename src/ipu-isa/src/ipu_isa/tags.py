"""Where a slot or instruction exists: in silicon, in the emulator, or only in
the assembler (E03-07, decision Q17).

Every slot, instruction and pseudo-instruction has exactly one tag. A slot is
HARDWARE or EMULATOR; an instruction takes its slot's tag; a
pseudo-instruction is ASSEMBLER (it expands to real instructions and has no
opcode). The SystemVerilog package and the RTL docs describe the HARDWARE
part only (E03-22, E03-23); the emulator word carries every slot (Q41).
"""

from __future__ import annotations

from enum import Enum


class Tag(str, Enum):
    """A tag. ``Tag.HARDWARE == "hardware"``.

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Slot``.
    """

    HARDWARE = "hardware"  # implemented by the IPU
    EMULATOR = "emulator"  # exists only in the emulator; never in the hardware word
    ASSEMBLER = "assembler"  # expanded by the assembler; no opcode

    def __str__(self) -> str:
        return self.value
