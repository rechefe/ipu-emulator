"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ipu_emu.regfile import RegFile
from ipu_emu.xmem import XMem

# Matches C: #define IPU__INST_MEM_SIZE 1024
INST_MEM_SIZE = 1024

# CR register index for data type
CR_DTYPE_REG = 15


class IpuState:
    """Complete IPU processor state.

    Attributes:
        regfile:         The live register file.
        xmem:            External memory (2 MB).
        program_counter: Current instruction address.
        inst_mem:        Instruction memory (list of decoded instruction dicts).
    """

    def __init__(self) -> None:
        self.regfile = RegFile()
        self.xmem = XMem()
        self.program_counter: int = 0
        # Instruction memory — each entry will be a decoded instruction dict
        # (populated when loading a binary or assembling).
        self.inst_mem: list[dict[str, Any] | None] = [None] * INST_MEM_SIZE

    # -- CR dtype convenience (mirrors ipu__set_cr_dtype / ipu__get_cr_dtype) --

    def set_cr_dtype(self, dtype: int) -> None:
        """Set the data type register (CR[15])."""
        self.regfile.set_cr(CR_DTYPE_REG, dtype)

    def get_cr_dtype(self) -> int:
        """Read the data type register (CR[15])."""
        return self.regfile.get_cr(CR_DTYPE_REG)

    # -- register file snapshot (for VLIW dispatch) -------------------------

    def snapshot_regfile(self) -> RegFile:
        """Deep-copy the register file for VLIW read-before-write semantics."""
        return self.regfile.snapshot()

    # -- XMEM ↔ register transfers (mirrors ipu__load_r_reg / ipu__store_r_reg) --

    def load_r_reg_from_xmem(self, xmem_addr: int, r_index: int) -> None:
        """Load 128 bytes from XMEM into R register *r_index*."""
        data = self.xmem.read_address(xmem_addr, 128)
        self.regfile.set_r(r_index, data)

    def store_r_reg_to_xmem(self, xmem_addr: int, r_index: int) -> None:
        """Store R register *r_index* (128 bytes) to XMEM."""
        data = self.regfile.get_r(r_index)
        self.xmem.write_address(xmem_addr, data)

    def load_r_cyclic_from_xmem(self, xmem_addr: int) -> None:
        """Load 128 bytes from XMEM into the cyclic register at current index."""
        data = self.xmem.read_address(xmem_addr, 128)
        # Load into the start of the cyclic register
        self.regfile.set_r_cyclic_at(0, data)

    def store_acc_to_xmem(self, xmem_addr: int) -> None:
        """Store the accumulator (512 bytes) to XMEM."""
        data = self.regfile.get_r_acc_bytes()
        self.xmem.write_address(xmem_addr, data)

    def load_r_mask_from_xmem(self, xmem_addr: int) -> None:
        """Load 128 bytes from XMEM into the mask register."""
        data = self.xmem.read_address(xmem_addr, 128)
        self.regfile.set_r_mask(data)

    # -- state queries ------------------------------------------------------

    @property
    def is_halted(self) -> bool:
        """True if PC has run past instruction memory."""
        return self.program_counter >= INST_MEM_SIZE

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full IPU state for debug/JSON export."""
        return {
            "program_counter": self.program_counter,
            "regfile": self.regfile.to_dict(),
        }

    def __repr__(self) -> str:
        return f"IpuState(pc={self.program_counter}, halted={self.is_halted})"
