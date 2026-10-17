"""BREAK slot handlers (``break`` is a keyword, hence ``brk``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.outcome import CycleOutcome

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu


def execute_break_nop(ipu: Ipu) -> CycleOutcome:
    """Execute NOP in break slot: No operation."""
    return CycleOutcome.RETIRED


def execute_break(ipu: Ipu) -> CycleOutcome:
    """Execute BREAK: Unconditional break."""
    return CycleOutcome.BREAK


def execute_break_ifeq(ipu: Ipu, *, reg: int, value: int) -> CycleOutcome:
    """Execute BREAK.IFEQ: Break if LR register equals immediate."""
    if reg == value:
        return CycleOutcome.BREAK
    return CycleOutcome.RETIRED
