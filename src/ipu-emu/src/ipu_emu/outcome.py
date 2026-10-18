"""How one call into the cycle ended: :class:`CycleOutcome`.

Its own module so the BREAK slot's handlers (``ipu_emu.slots.brk``) can return
it without importing the cycle module, which imports the engine's plan tables
and through them every slot module. ``ipu_emu.cycle`` re-exports it.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = ["CycleOutcome"]


class CycleOutcome(IntEnum):
    """How one call into the cycle ended."""

    RETIRED = 0
    BREAK = 1
