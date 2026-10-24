"""Which programs a run mode accepts (ADR 0007).

``ACTIVATE.EXACT`` stores an FP32 row, which NX8 has no room for, so an
NX8-mode state refuses a program that uses it: when the program is loaded
(``emulator.load_program``) and when a state holding it is reset to NX8
(``IpuState.reset``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ipu_emu.errors import ConfigurationError
from ipu_isa.codec.word import opcode_field
from ipu_isa.formats import ElementFormat
from ipu_isa.slots import Slot
from ipu_isa.view import isa

__all__ = ["check_program_mode"]

#: Instructions valid in FP32 mode only (ADR 0007): (opcode field, opcode, mnemonic).
_FP32_ONLY = tuple(
    (opcode_field(slot), isa().instruction(slot, mnemonic).opcode, mnemonic)
    for slot, mnemonic in ((Slot.AAQ_EXACT, "ACTIVATE.EXACT"),)
)


def check_program_mode(fmt: ElementFormat, instructions: Sequence[Any]) -> None:
    """Refuse a program that uses an FP32-only instruction in an NX8-mode state.

    ``ACTIVATE.EXACT`` stores an FP32 row, which NX8 has no room for; the
    check runs when the program is loaded and when a state is reset to NX8
    (ADR 0007).

    Raises:
        ConfigurationError: *fmt* is NX8 and a word holds ``ACTIVATE.EXACT``.
    """
    if fmt is not ElementFormat.NX8:
        return
    for pc, inst in enumerate(instructions):
        for field, opcode, mnemonic in _FP32_ONLY:
            if inst is not None and inst.get(field) == opcode:
                raise ConfigurationError(
                    f"word {pc} uses {mnemonic}, which is valid in FP32 mode only; "
                    "this state runs in NX8 mode (ADR 0007)"
                )
