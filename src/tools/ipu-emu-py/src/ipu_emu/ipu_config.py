"""Emulator settings that are conventions, not ISA facts (E03-12).

The ISA constants (CR0/CR1, register widths, the dstructure layout) live in
``ipu_isa.constants``.
"""

from __future__ import annotations

#: The CR the emulator and the kernels keep the dstructure configuration in.
#: A convention: any CR0-CR15 is a valid ISA operand, CR15 included.
DEFAULT_DSTRUCTURE_CR = 15
