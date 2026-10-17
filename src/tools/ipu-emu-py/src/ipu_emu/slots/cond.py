"""COND slot handlers: branches, HALT, and the PC advance of its NOP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.ipu_state import INST_MEM_SIZE, HaltRecord
from ipu_isa.constants import LR_CR_SCALAR_BITS

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu


def execute_beq(ipu: Ipu, *, reg1: int, reg2: int, label: int) -> None:
    """Execute BEQ: Branch if equal."""
    ipu.state.program_counter = label if reg1 == reg2 else ipu.state.program_counter + 1


def execute_bne(ipu: Ipu, *, reg1: int, reg2: int, label: int) -> None:
    """Execute BNE: Branch if not equal."""
    ipu.state.program_counter = label if reg1 != reg2 else ipu.state.program_counter + 1


def _to_signed_reg(value: int) -> int:
    """Sign-extend a value at the LR/CR register width (32 bits)."""
    if value >= (1 << (LR_CR_SCALAR_BITS - 1)):
        return value - (1 << LR_CR_SCALAR_BITS)
    return value


def execute_blt_s(ipu: Ipu, *, reg1: int, reg2: int, label: int) -> None:
    """Execute BLT.S: Branch if less than (signed comparison)."""
    s1 = _to_signed_reg(reg1)
    s2 = _to_signed_reg(reg2)
    ipu.state.program_counter = label if s1 < s2 else ipu.state.program_counter + 1


def execute_bge_s(ipu: Ipu, *, reg1: int, reg2: int, label: int) -> None:
    """Execute BGE.S: Branch if greater or equal (signed comparison)."""
    s1 = _to_signed_reg(reg1)
    s2 = _to_signed_reg(reg2)
    ipu.state.program_counter = label if s1 >= s2 else ipu.state.program_counter + 1


def execute_br(ipu: Ipu, *, reg: int) -> None:
    """Execute BR: Branch to register value."""
    ipu.state.program_counter = reg


def execute_halt(ipu: Ipu) -> None:
    """Execute HALT: record the halt at this word, then move the PC past
    IMEM. COND runs last, so the rest of the word has retired."""
    ipu.state.halt = HaltRecord("halt", ipu.state.program_counter, None)
    ipu.state.program_counter = INST_MEM_SIZE


def execute_cond_nop(ipu: Ipu) -> None:
    """Execute NOP in cond slot: No operation; advance PC."""
    ipu.state.program_counter += 1
