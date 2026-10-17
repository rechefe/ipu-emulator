"""LOAD slot handlers: XMEM rows into the multiply stage, R_CYCLIC and R_MASK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.errors import EmulatorFault
from ipu_emu.geometry import xmem_element_width_bytes, xmem_row_addr, xmem_row_size_bytes
from ipu_isa.faults import FaultId
from ipu_isa.registers import get_mult_stage_map, get_register_sizes

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu

_reg_sizes = get_register_sizes()
R_REG_SIZE = _reg_sizes["r"]["size_bytes"]
R_CYCLIC_SIZE = _reg_sizes["r_cyclic"]["size_bytes"]

# MultStageRegField: index → (register_name, element_index)
# e.g. [("r", 0), ("r", 1)] — encoding 2 is reserved / invalid in assembly
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()

# R_CYCLIC is divided into four 128-byte slots; LD.MULT.CYCLIC's index
# must land exactly on a slot boundary — no implicit wraparound.
R_CYCLIC_VALID_INDICES = tuple(range(0, R_CYCLIC_SIZE, R_REG_SIZE))


def execute_load_nop(ipu: Ipu) -> None:
    """Execute NOP in load slot: No operation."""


def execute_ld_mult(ipu: Ipu, *, dest: int, offset: int, base: int) -> None:
    """Execute LD.MULT: Load data from memory into a mult stage register."""
    if dest not in (0, 1):
        raise EmulatorFault(
            FaultId.F1,
            ipu.state.program_counter,
            f"LD.MULT: dest must be 0 (r0) or 1 (r1); got {dest}",
        )
    state = ipu.state
    addr = xmem_row_addr(state, offset + base)
    if state.wide_vector_debug:
        data = state.xmem.read_address(addr, xmem_row_size_bytes(state))
        state.regfile.set_r_wide_debug(dest, data)
        if state.xmem.is_constant_ones(addr, data):
            state.regfile.mark_constant_ones("r_wide_debug", dest * len(data), len(data))
        return

    data = state.xmem.read_address(addr, R_REG_SIZE)
    reg_name, elem_idx = _MULT_STAGE_MAP[dest]
    state.regfile.set_register_bytes(reg_name, elem_idx, data)
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(reg_name, elem_idx * len(data), len(data))


def execute_ld_mult_cyclic(ipu: Ipu, *, offset: int, base: int, index: int) -> None:
    """Execute LD.MULT.CYCLIC: Load with cyclic addressing into r_cyclic.

    ``index`` is an ELEMENT index into r_cyclic's 512-element ring and must
    land on one of the four slot boundaries (0/128/256/384) in both modes
    -- writes only ever replace a whole slot, never a partial one. This is
    unlike reads (MULT.RC.* via ``get_r_cyclic_wide_debug_at``), which are
    allowed at any element index and may cross a slot boundary.
    """
    state = ipu.state
    addr = xmem_row_addr(state, offset + base)
    if index not in R_CYCLIC_VALID_INDICES:
        raise EmulatorFault(
            FaultId.F4,
            state.program_counter,
            f"LD.MULT.CYCLIC: index must be one of {R_CYCLIC_VALID_INDICES} "
            f"(R_CYCLIC slot boundaries); got {index}",
        )
    byte_idx = index * xmem_element_width_bytes(state)
    data = state.xmem.read_address(addr, xmem_row_size_bytes(state))
    if state.wide_vector_debug:
        state.regfile.set_r_cyclic_wide_debug_at(byte_idx, data)
    else:
        state.regfile.set_r_cyclic_at(byte_idx, data)
    if state.xmem.is_constant_ones(addr, data):
        name = "r_cyclic_wide_debug" if state.wide_vector_debug else "r_cyclic"
        state.regfile.mark_constant_ones(name, byte_idx, len(data))


def execute_ld_mask(ipu: Ipu, *, offset: int, base: int) -> None:
    """Execute LD.MASK: Load mask data from memory.

    Reads only the START of the row -- 128 bytes (8 x 128-bit slots) in
    both modes. The mask is 1 bit per element and does not scale with
    element width, so only the row address is translated; the read size
    stays R_REG_SIZE regardless of mode.
    """
    addr = xmem_row_addr(ipu.state, offset + base)
    data = ipu.state.xmem.read_address(addr, R_REG_SIZE)
    ipu.state.regfile.set_r_mask(data)
