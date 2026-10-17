"""LOAD slot handlers: XMEM rows into the multiply stage, R_CYCLIC and R_MASK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import vector_registers, xmem_row_addr
from ipu_isa.geometry import LANES
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


def _row_bytes(ipu: Ipu) -> int:
    """Bytes of one XMEM row in the active format: LANES elements."""
    return ipu._codec.element_bytes * LANES


def execute_ld_mult(ipu: Ipu, *, dest: int, offset: int, base: int) -> None:
    """Execute LD.MULT: Load one XMEM row into R0 or R1 (Ra in the active element width)."""
    if dest not in (0, 1):
        raise EmulatorError(f"LD.MULT: dest must be 0 (r0) or 1 (r1); got {dest}")
    state = ipu.state
    addr = xmem_row_addr(state, offset + base)
    data = state.xmem.read_address(addr, _row_bytes(ipu))
    reg_name = vector_registers(state).ra
    elem_idx = _MULT_STAGE_MAP[dest][1]
    state.regfile.set_register_bytes(reg_name, elem_idx, data)
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(reg_name, elem_idx * len(data), len(data))


def execute_ld_mult_cyclic(ipu: Ipu, *, offset: int, base: int, index: int) -> None:
    """Execute LD.MULT.CYCLIC: Load with cyclic addressing into r_cyclic.

    ``index`` is an ELEMENT index into r_cyclic's 512-element ring and must
    land on one of the four slot boundaries (0/128/256/384) in every format
    -- writes only ever replace a whole slot, never a partial one. This is
    unlike reads (MULT.RC.*), which are allowed at any element index and may
    cross a slot boundary.
    """
    state = ipu.state
    addr = xmem_row_addr(state, offset + base)
    if index not in R_CYCLIC_VALID_INDICES:
        raise EmulatorError(
            f"LD.MULT.CYCLIC: index must be one of {R_CYCLIC_VALID_INDICES} "
            f"(R_CYCLIC slot boundaries); got {index}"
        )
    row_bytes = _row_bytes(ipu)
    byte_idx = index * ipu._codec.element_bytes
    data = state.xmem.read_address(addr, row_bytes)
    name = vector_registers(state).r_cyclic
    # A slot never wraps the ring, so this is the ring's own slot write.
    state.regfile.mutable(name, byte_idx, row_bytes)[:] = data
    if state.xmem.is_constant_ones(addr, data):
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
