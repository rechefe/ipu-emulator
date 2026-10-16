"""The repository ISA's derived binary layout (E03-07, E03-08, E03-09).

:func:`isa_layout` is ``isa().layout``: the frozen :class:`IsaLayout` that
:func:`ipu_isa.union_layout.compute_layout` solved when the ISA view was
loaded. Every slot's packed fields, and the width of every operand type,
including the widths only packing decides (``LrIncDecImmediate``,
``LrOrReshapeMaskImmediate``). Nothing is written into other modules, so the
result does not depend on import order.
"""

from __future__ import annotations

from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.union_layout import IsaLayout
from ipu_isa.view import isa


def isa_layout() -> IsaLayout:
    """The layout of the repository's ISA."""
    return isa().layout


def slots_with_tag(tag: Tag) -> tuple[Slot, ...]:
    """The repository's slots tagged ``tag``, in layout order."""
    return isa_layout().slots_with_tag(tag)


def word_bits(tag: Tag | None = None) -> int:
    """Width of the emulator word (``None``) or of its ``tag`` slots."""
    return isa_layout().word_bits(tag)
