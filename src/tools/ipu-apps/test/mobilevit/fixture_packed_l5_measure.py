"""Measurement helpers for the L5 packed-layout viability experiment.

Standalone test support (not a shipped ipu_apps kernel module) -- uses the
direct assemble_to_bin_file + IpuState pattern. Builds only the packed
kernels; never imports or runs residual_add_16x240, qk_scores_16x60,
attn_v_16x60, or any softmax kernel.

Instruction counting: RunStats (ipu_emu/stats.py) tracks cycles and
mult/acc-active cycles and XMEM accesses, but NOT a dynamic executed-
instruction count. This counts every non-NOP op the cycle issues (LR slots
aside), the BREAK slot included, as a CycleObserver attached to the state
(IpuState.add_cycle_observer) for the duration of one `with` block. The cycle
issues the same non-NOP ops in faithful and prototyping mode (prototyping only
skips NOP slots), so the counts are identical in both modes. Nothing in Ipu is
patched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ipu_emu.ipu_state import IpuState
from ipu_emu.issue import IssueFacts
from ipu_isa.slots import Slot

#: The slots counted: every slot but LR.
_COUNTED = frozenset(slot.value for slot in Slot if slot is not Slot.LR)


@dataclass
class InstrCount:
    """Dynamic executed non-NOP instruction count, broken down by slot.

    A :class:`~ipu_emu.cycle.CycleObserver`: it counts ``issued`` events.
    """

    by_slot: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_slot.values())

    def begin(self, pc: int) -> None:
        pass

    def issued(self, facts: IssueFacts) -> None:
        if facts.slot in _COUNTED and facts.plan.inst_name != "NOP":
            self.by_slot[facts.slot] = self.by_slot.get(facts.slot, 0) + 1

    def retired(self, facts: IssueFacts, result: Any) -> None:
        pass

    def end(self) -> None:
        pass


@contextmanager
def count_instructions(state: IpuState) -> Iterator[InstrCount]:
    """Context manager: yields an InstrCount that fills in as *state* runs.

    The count observes every cycle on *state* for the duration of the `with`
    block only; it is detached on exit regardless of how the block ends.
    """
    counts = InstrCount()
    remove = state.add_cycle_observer(counts)
    try:
        yield counts
    finally:
        remove()
