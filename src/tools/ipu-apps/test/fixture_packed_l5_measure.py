"""Measurement helpers for the L5 packed-layout viability experiment.

Standalone, throwaway (not a shipped ipu_apps kernel module) -- follows the
same direct assemble_to_bin_file + IpuState pattern as
test_decisive_l5_uncropped.py. Builds NEW kernels only; never imports or
runs residual_add_16x240, qk_scores_16x60, attn_v_16x60, or any softmax
kernel.

Instruction counting: RunStats (ipu_emu/stats.py) tracks cycles and
mult/acc-active cycles and XMEM accesses, but NOT a dynamic executed-
instruction count. Rather than reimplement NOP detection (risk of drifting
from the real logic and fabricating a number), this monkeypatches
Ipu.dispatch_instruction -- the exact seam where the emulator itself
resolves "instruction_name != NOP" (ipu.py:1343, see the mult/acc/load/store
stats updates a few lines below the lookup) -- to tally every dispatched,
non-NOP slot across every executed cycle. No ipu.py edit; this only wraps
the method from outside for the duration of one run() call.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from ipu_emu.ipu import Ipu


@dataclass
class InstrCount:
    """Dynamic executed non-NOP instruction count, broken down by slot."""

    by_slot: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.by_slot.values())


@contextmanager
def count_instructions():
    """Context manager: yields an InstrCount that fills in as the emulator runs.

    Wraps Ipu.dispatch_instruction for the duration of the `with` block only;
    the original method is restored on exit regardless of how the block ends.
    """
    counts = InstrCount()
    original = Ipu.dispatch_instruction

    # We need the resolved instruction_name, which `original` computes
    # internally but does not return. Re-derive it the same way
    # dispatch_instruction does (get_instruction_by_opcode), rather than
    # guessing from the return value -- this stays byte-for-byte aligned
    # with the emulator's own NOP classification.
    from ipu_common.instruction_spec import get_instruction_by_opcode
    from ipu_emu.ipu import _SLOT_FIELD_PREFIX

    def counting_dispatch(self, slot_type, inst):
        prefix = _SLOT_FIELD_PREFIX[slot_type]
        opcode_field = f"{prefix}_token_0_{slot_type}_inst_opcode"
        opcode = inst[opcode_field]
        instruction_name, _spec = get_instruction_by_opcode(slot_type, opcode)
        if instruction_name != "NOP":
            counts.by_slot[slot_type] = counts.by_slot.get(slot_type, 0) + 1
        return original(self, slot_type, inst)

    Ipu.dispatch_instruction = counting_dispatch
    try:
        yield counts
    finally:
        Ipu.dispatch_instruction = original
