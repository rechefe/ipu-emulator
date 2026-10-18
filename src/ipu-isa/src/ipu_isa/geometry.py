"""Sizes of the machine that the ISA encodes against (E03-05, E03-12).

Each size is defined once here; the assembler, emulator, harness and kernels
import it instead of restating it. Sizes that depend on the element format
(element and row bytes) come with the element formats (E03-13).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ipu_isa.registers import REGISTER_DEFINITIONS, get_lane_count
from ipu_isa.tags import Tag

#: SIMD lanes in the datapath (``r_acc``'s word view).
LANES: Final[int] = get_lane_count()

#: Instruction-memory depth in words. A program has at most this many
#: instruction words, so a branch target needs ``(INST_MEM_WORDS - 1).bit_length()``
#: bits. Requirement OQ-4 leaves making it a run-time parameter open.
INST_MEM_WORDS: Final[int] = 1024

#: Elements in the cyclic register ``r_cyclic`` (one byte each).
R_CYCLIC_ELEMENTS: Final[int] = REGISTER_DEFINITIONS["r_cyclic"]["size_bytes"]

#: The emulator's XMEM address space in bytes (512 MiB, allocated lazily).
#: Not a hardware size: see ``TAGS``.
XMEM_BYTES: Final[int] = 1 << 29

#: Where each size holds: in the IPU, or only in the emulator.
TAGS: Mapping[str, Tag] = MappingProxyType(
    {
        "LANES": Tag.HARDWARE,
        "INST_MEM_WORDS": Tag.HARDWARE,
        "R_CYCLIC_ELEMENTS": Tag.HARDWARE,
        "XMEM_BYTES": Tag.EMULATOR,
    }
)
