"""Sizes of the machine that the ISA encodes against.

E03-12 adds the rest of the geometry (lanes, row bytes, element widths) here.
"""

from __future__ import annotations

#: Instruction-memory depth: a program has at most this many instruction words,
#: so a branch target needs ``(MAX_PROGRAM_SIZE - 1).bit_length()`` bits.
MAX_PROGRAM_SIZE = 1024
