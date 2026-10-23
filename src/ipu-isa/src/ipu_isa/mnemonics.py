"""The mnemonic standard (E03-19, Q33/Q45), checked by the spec schema.

* A mnemonic is upper-case words joined by dots: ``[A-Z][A-Z0-9]*(\\.[A-Z0-9]+)*``.
  No underscore.
* Control-stage instructions (LR, COND, BREAK) use plain RISC-style names with
  variants as dot suffixes (``BLT.S``, ``INC.MODP2``).
* Every instruction of a datapath slot (LOAD, MULT, ACC, AAQ, AAQ_EXACT)
  but ``NOP`` is ``UNIT.OP[.VARIANT]``, so it contains a dot.
* Branch signedness is in the name. The unsigned branches are reserved: no
  spec may define them, and the assembler rejects them as "reserved, not
  implemented".
"""

from __future__ import annotations

import re

from ipu_isa.schema import SpecError
from ipu_isa.slots import Slot

#: The form of every mnemonic, real or pseudo.
MNEMONIC_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(\.[A-Z0-9]+)*")

#: Slots whose instructions are named ``UNIT.OP[.VARIANT]``.
DATAPATH_SLOTS = frozenset({Slot.LOAD, Slot.MULT, Slot.ACC, Slot.AAQ, Slot.AAQ_EXACT})

#: Names kept for instructions not implemented yet (Q33): no spec defines them.
RESERVED_MNEMONICS = frozenset({"BLT.U", "BGE.U", "BGT.U", "BLE.U"})


def check_mnemonic(path: str, mnemonic: str, slot: Slot | None) -> None:
    """Raise ``SpecError`` if ``mnemonic`` breaks the standard.

    Args:
        path: Where the mnemonic is defined, for the message.
        mnemonic: The name as written in the spec.
        slot: The instruction's slot; ``None`` for a pseudo-instruction.

    Raises:
        SpecError: It does not match ``MNEMONIC_PATTERN``, is reserved, or is a
            datapath instruction without a dot.
    """
    if not MNEMONIC_PATTERN.fullmatch(mnemonic):
        raise SpecError(
            f"{path}: mnemonic {mnemonic!r} must be upper-case words joined by dots "
            f"({MNEMONIC_PATTERN.pattern}), no underscore"
        )
    if mnemonic in RESERVED_MNEMONICS:
        raise SpecError(f"{path}: mnemonic {mnemonic!r} is reserved (Q33) and may not be defined")
    if slot in DATAPATH_SLOTS and mnemonic != "NOP" and "." not in mnemonic:
        raise SpecError(
            f"{path}: datapath mnemonic {mnemonic!r} must be UNIT.OP[.VARIANT] (contain a dot)"
        )
