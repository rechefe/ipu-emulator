"""Measurement helpers for the L5 packed-layout viability experiment.

Standalone test support (not a shipped ipu_apps kernel module) -- uses the
direct assemble_to_bin_file + IpuState pattern. Builds only the packed
kernels; never imports or runs residual_add_16x240, qk_scores_16x60,
attn_v_16x60, or any softmax kernel.

Instruction counting: RunStats (ipu_emu/stats.py) tracks cycles and
mult/acc-active cycles and XMEM accesses, but NOT a dynamic executed-
instruction count. This counts every non-NOP slot of every executed cycle
(LR slots aside), as the emulator itself classifies the opcode
(isa().by_opcode). It wraps the cycle entry points,
Ipu.execute_vliw_cycle and Ipu.execute_vliw_cycle_skip_break, which both the
faithful path and prototyping mode's fast path run through -- the fast path
skips Ipu.dispatch_instruction, so counting there read zero in that mode. The
counts match the per-dispatch ones: a cycle's break slot counts when the cycle
starts, its other slots once it continues past the break. No ipu.py edit; the
methods are wrapped from outside for the duration of one `with` block.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from ipu_emu.execute import BreakResult
from ipu_emu.ipu import _SLOT_FIELD_PREFIX, Ipu
from ipu_isa.slots import Slot
from ipu_isa.view import isa

#: The slots a cycle runs after its break slot (the LR slots are not counted).
_AFTER_BREAK = tuple(slot for slot in _SLOT_FIELD_PREFIX if slot != "break")


@dataclass
class InstrCount:
    """Dynamic executed non-NOP instruction count, broken down by slot."""

    by_slot: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_slot.values())

    def add(self, inst: dict | None, slots) -> None:
        """Count the non-NOP instructions of *inst* in *slots*."""
        if inst is None:
            return
        for slot_type in slots:
            opcode = inst[f"{_SLOT_FIELD_PREFIX[slot_type]}_token_0_{slot_type}_inst_opcode"]
            if isa().by_opcode(Slot(slot_type), opcode).mnemonic != "NOP":
                self.by_slot[slot_type] = self.by_slot.get(slot_type, 0) + 1


@contextmanager
def count_instructions():
    """Context manager: yields an InstrCount that fills in as the emulator runs.

    Wraps Ipu.execute_vliw_cycle and Ipu.execute_vliw_cycle_skip_break for the
    duration of the `with` block only; the originals are restored on exit
    regardless of how the block ends.
    """
    counts = InstrCount()
    cycle, rest_of_cycle = Ipu.execute_vliw_cycle, Ipu.execute_vliw_cycle_skip_break

    def counting_cycle(self):
        inst = self.state.inst_mem[self.state.program_counter]
        counts.add(inst, ("break",))
        result = cycle(self)
        if result != BreakResult.BREAK:
            counts.add(inst, _AFTER_BREAK)
        return result

    def counting_rest_of_cycle(self):
        counts.add(self.state.inst_mem[self.state.program_counter], _AFTER_BREAK)
        return rest_of_cycle(self)

    Ipu.execute_vliw_cycle = counting_cycle
    Ipu.execute_vliw_cycle_skip_break = counting_rest_of_cycle
    try:
        yield counts
    finally:
        Ipu.execute_vliw_cycle = cycle
        Ipu.execute_vliw_cycle_skip_break = rest_of_cycle
