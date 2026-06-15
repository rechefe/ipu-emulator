"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem
from ipu_common import activations as _activations
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import (
    CR_DSTRUCTURE_REG_INDEX,
    DEFAULT_DSTRUCTURE,
    DStructureConfig,
    Partition,
    decode_dstructure,
    encode_dstructure,
)

# Matches C: #define IPU__INST_MEM_SIZE 1024
INST_MEM_SIZE = 1024

class WideVectorArithmetic(str, Enum):
    """How 128-lane wide-vector debug math is performed (emulator-only; issue #33).

    FP32: each lane is IEEE float32 (default for "no quantization" FP analysis).
    INT32: each lane is signed int32 with wrap semantics matching INT8-mode acc ops.
    """

    FP32 = "fp32"
    INT32 = "int32"


class IpuState:
    """Complete IPU processor state.

    Attributes:
        regfile:         The live register file.
        xmem:            External memory (2 MB).
        program_counter: Current instruction address.
        inst_mem:        Instruction memory (list of decoded instruction dicts).
        dtype:           Arithmetic data type (not stored in CR; emulator-only).
    """

    def __init__(
        self,
        *,
        wide_vector_debug: bool = False,
        wide_vector_arithmetic: WideVectorArithmetic = WideVectorArithmetic.FP32,
        wide_vector_quantize_output: bool = False,
        elu_alpha: float | None = None,
        dtype: DType = DType.INT8,
    ) -> None:
        self.regfile = RegFile()
        self.xmem = XMem()
        self.program_counter: int = 0
        self.stats = RunStats()
        self.inst_mem: list[dict[str, Any] | None] = [None] * INST_MEM_SIZE

        # Arithmetic data type — not stored in a CR register (emulator-only).
        self.dtype: DType = dtype

        self.set_cr_dstructure(
            valid_elements=DEFAULT_DSTRUCTURE.valid_elements,
            partition=DEFAULT_DSTRUCTURE.partition,
        )

        # --- Emulator-only wide-vector debug mode (GitHub issue #33) ------------
        self.wide_vector_debug: bool = wide_vector_debug
        self.wide_vector_arithmetic: WideVectorArithmetic = wide_vector_arithmetic
        self.wide_vector_quantize_output: bool = wide_vector_quantize_output
        self._debug_mult_stage_vectors: dict[int, list[float | int]] = {}
        self._debug_mult_stage_vectors_snap: dict[int, list[float | int]] = {}

        # --- Activation α (emulator-only; not mapped to CR) ----------------------
        self.elu_alpha: float = (
            float(elu_alpha) if elu_alpha is not None else float(_activations._ELU_ALPHA)
        )

    # -- CR dstructure convenience (CR15 = valid_elements[7:0] | partition[11:8]) --

    def get_cr_dstructure(self) -> DStructureConfig:
        """Read CR15 as decoded dstructure configuration fields."""
        return decode_dstructure(self.regfile.get_cr(CR_DSTRUCTURE_REG_INDEX))

    def get_config_valid_elements(self) -> int:
        """Return the active lane count from the CR15 dstructure register."""
        return self.get_cr_dstructure().valid_elements

    def get_config_partition(self) -> int:
        """Return the partition field from the CR15 dstructure register."""
        return self.get_cr_dstructure().partition

    def set_cr_dstructure(
        self,
        valid_elements: int = DEFAULT_DSTRUCTURE.valid_elements,
        partition: Partition | int = DEFAULT_DSTRUCTURE.partition,
    ) -> None:
        """Write CR15 as dstructure configuration fields."""
        self.regfile.set_cr(
            CR_DSTRUCTURE_REG_INDEX,
            encode_dstructure(valid_elements=valid_elements, partition=partition),
        )

    def set_activation_alphas(
        self,
        *,
        elu_alpha: float | None = None,
    ) -> None:
        """Override α for ``elu`` (emulator-only; not CR).

        Only arguments that are not ``None`` are updated. Values apply to subsequent
        ``ACTIVATE`` instructions executed on this state.
        """
        if elu_alpha is not None:
            self.elu_alpha = float(elu_alpha)

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
