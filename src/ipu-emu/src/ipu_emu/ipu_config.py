"""Emulator settings that are conventions, not ISA facts (E03-12).

The ISA constants (CR0/CR1, register widths) live in ``ipu_isa.constants``,
the DS, QF and constant layouts in ``ipu_isa.cr_halves``.
"""

from __future__ import annotations

#: The CR half the emulator and the kernels keep the default DS in: ``CR15.L``
#: (ADR 0007). A convention: any half of CR0-CR15 is a valid DS operand.
DEFAULT_DS_CR = 15
DEFAULT_DS_HIGH = False
