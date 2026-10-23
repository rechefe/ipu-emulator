"""LOAD slot handlers: XMEM rows into the multiply stage, R_CYCLIC and R_MASK.

A row is 128 B in NX8 mode and 512 B in FP32 mode (ADR 0007). ``LD.MULT`` and
``LD.MULT.CYCLIC`` copy the row's metadata word into ``r_meta`` /
``r_cyclic_meta`` beside it; in FP32 mode a row has no metadata and they
write 0. ``LD.MASK`` reads the first 128 B in both modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ipu_emu.geometry import (
    R_CYCLIC_VALID_INDICES,
    R_REG_SIZE,
    fp32_mode,
    vector_registers,
    xmem_row_addr,
    xmem_row_check,
    xmem_row_size_bytes,
)
from ipu_isa.faults import FaultId
from ipu_isa.registers import get_mult_stage_map

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu

# MultStageRegField: index → (register_name, element_index)
# e.g. [("r", 0), ("r", 1)] — encoding 2 is reserved / invalid in assembly
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()


#: Bytes of one metadata register entry (``r_meta``, ``r_cyclic_meta``).
_META_BYTES = 2


def _row_metadata(ipu: Ipu, addr: int) -> bytes:
    """The metadata a load copies beside the row at *addr*: the XMEM side
    store's word in NX8 mode, 0 in FP32 mode (a 512-byte row has none)."""
    state = ipu.state
    word = 0 if fp32_mode(state) else state.xmem.read_metadata(addr)
    return word.to_bytes(_META_BYTES, "little")


def execute_load_nop(ipu: Ipu) -> None:
    """Execute NOP in load slot: No operation."""


Fault = tuple[FaultId, str]


def check_ld_mult(ipu: Ipu, *, dest: int, offset: int, base: int) -> Fault | None:
    """F-6 for a row outside XMEM (``dest`` other than R0/R1 is F-1 at fetch)."""
    return xmem_row_check(ipu.state, offset + base)


def _row_bytes(ipu: Ipu) -> int:
    """Bytes of one XMEM row in the active format: LANES elements. The row
    size the F-6 check uses (``xmem_row_size_bytes``) is the same number."""
    return xmem_row_size_bytes(ipu.state)


def execute_ld_mult(ipu: Ipu, *, dest: int, offset: int, base: int) -> None:
    """Execute LD.MULT: Load one XMEM row into R0 or R1 (Ra in the active element
    width), and its metadata into ``r_meta``."""
    state = ipu.state
    addr = xmem_row_addr(state, offset + base)
    data = state.xmem.read_address(addr, _row_bytes(ipu))
    reg_name = vector_registers(state).ra
    elem_idx = _MULT_STAGE_MAP[dest][1]
    state.regfile.set_register_bytes(reg_name, elem_idx, data)
    state.regfile.set_register_bytes("r_meta", elem_idx, _row_metadata(ipu, addr))
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(reg_name, elem_idx * len(data), len(data))


def check_ld_mult_cyclic(ipu: Ipu, *, offset: int, base: int, index: int) -> Fault | None:
    """F-4 for an index off a slot boundary, else F-6 for a row outside XMEM
    (the lower number first: IPU-FLT-8)."""
    if index not in R_CYCLIC_VALID_INDICES:
        return (
            FaultId.F4,
            f"LD.MULT.CYCLIC: index must be one of {R_CYCLIC_VALID_INDICES} "
            f"(R_CYCLIC slot boundaries); got {index}",
        )
    return xmem_row_check(ipu.state, offset + base)


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
    row_bytes = _row_bytes(ipu)
    byte_idx = index * ipu.codec.element_bytes
    data = state.xmem.read_address(addr, row_bytes)
    name = vector_registers(state).r_cyclic
    # A slot never wraps the ring, so this is the ring's own slot write.
    state.regfile.mutable(name, byte_idx, row_bytes)[:] = data
    slot = index // R_REG_SIZE  # R_CYCLIC holds four 128-element slots
    state.regfile.set_register_bytes("r_cyclic_meta", slot, _row_metadata(ipu, addr))
    if state.xmem.is_constant_ones(addr, data):
        state.regfile.mark_constant_ones(name, byte_idx, len(data))


def check_ld_mask(ipu: Ipu, *, offset: int, base: int) -> Fault | None:
    """F-6 for a row outside XMEM."""
    return xmem_row_check(ipu.state, offset + base)


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
