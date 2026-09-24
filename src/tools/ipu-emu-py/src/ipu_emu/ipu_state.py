"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

from enum import Enum
import struct
from typing import Any

from ipu_emu.prototyping import resolve_enabled as _resolve_prototyping
from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem
from ipu_common import activations as _activations
from ipu_emu.ipu_math import DType
from ipu_emu.ipu_config import (
    CR_DSTRUCTURE_REG_INDEX,
    DEFAULT_DSTRUCTURE,
    DStructureConfig,
    PadMode,
    Partition,
    decode_dstructure,
    encode_dstructure,
)

# Matches C: #define IPU__INST_MEM_SIZE 1024
INST_MEM_SIZE = 1024


def _ipu():
    """The ``ipu`` module, for its register sizes and row geometry.

    ``ipu`` imports this module, so importing it at load time would be circular.
    """
    from ipu_emu import ipu

    return ipu

class WideVectorArithmetic(str, Enum):
    """How 128-element wide-vector debug math is performed (emulator-only; issue #33).

    FP32: each element is IEEE float32 (default for "no quantization" FP analysis).
    INT32: each element is signed int32 with wrap semantics matching INT8-mode acc ops.
    """

    FP32 = "fp32"
    INT32 = "int32"


class IpuState:
    """Complete IPU processor state.

    Attributes:
        regfile:         The live register file.
        xmem:            External memory (512 MiB, mode-independent allocation).
        program_counter: Current instruction address.
        inst_mem:        Instruction memory (list of decoded instruction dicts).
        dtype:           Arithmetic data type (not stored in CR; emulator-only).
        prototyping:     Whether the opt-in fast paths are active (emulator-only).
    """

    def __init__(
        self,
        *,
        wide_vector_debug: bool = False,
        wide_vector_arithmetic: WideVectorArithmetic = WideVectorArithmetic.FP32,
        wide_vector_quantize_output: bool = False,
        elu_alpha: float | None = None,
        window_a: float | None = None,
        window_b: float | None = None,
        dtype: DType = DType.INT8,
        prototyping: bool | None = None,
        alias_profile=None,
    ) -> None:
        self.regfile = RegFile()
        self.xmem = XMem()
        self.program_counter: int = 0
        self.stats = RunStats()
        self.alias_profile = alias_profile
        self.stats.alias_profile = alias_profile
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

        # --- Prototyping mode (emulator-only; see ipu_emu/prototyping.py) --------
        # Opt-in fast paths that produce bit-identical state while skipping work
        # the hardware really does. The Bazel flag is a mandatory capability;
        # runtime controls may only select the mode inside a flagged build.
        self.prototyping: bool = _resolve_prototyping(prototyping)

        # --- Activation α (emulator-only; not mapped to CR) ----------------------
        self.elu_alpha: float = (
            float(elu_alpha) if elu_alpha is not None else float(_activations._ELU_ALPHA)
        )

        # --- Window bounds [a, b) for the ``window`` activation (emulator-only) ---
        self.window_a: float = (
            float(window_a) if window_a is not None else float(_activations.DEFAULT_WINDOW_A)
        )
        self.window_b: float = (
            float(window_b) if window_b is not None else float(_activations.DEFAULT_WINDOW_B)
        )

    # -- CR dstructure convenience (CR15 = valid_elements[7:0] | partition[11:8]) --

    def get_cr_dstructure(self) -> DStructureConfig:
        """Read CR15 as decoded dstructure configuration fields."""
        return decode_dstructure(self.regfile.get_cr(CR_DSTRUCTURE_REG_INDEX))

    def get_dstructure_for(self, cr_idx: int) -> DStructureConfig:
        """Decode the dstructure configuration from an arbitrary CR register.

        Lets mult/acc/aaq stage instructions select which CR supplies their
        valid element mask and dstructure info, instead of being hard-coded to CR15.
        """
        return decode_dstructure(self.regfile.get_cr(cr_idx))

    def set_cr_dstructure(
        self,
        valid_elements: int = DEFAULT_DSTRUCTURE.valid_elements,
        partition: Partition | int = DEFAULT_DSTRUCTURE.partition,
        pad_mode: PadMode | int = DEFAULT_DSTRUCTURE.pad_mode,
        *,
        cr_idx: int = CR_DSTRUCTURE_REG_INDEX,
    ) -> None:
        """Write dstructure configuration fields to ``cr_idx`` (CR15 by default)."""
        self.regfile.set_cr(
            cr_idx,
            encode_dstructure(valid_elements=valid_elements, partition=partition, pad_mode=pad_mode),
        )

    def set_activation_alphas(
        self,
        *,
        elu_alpha: float | None = None,
        window_a: float | None = None,
        window_b: float | None = None,
    ) -> None:
        """Override α for ``elu`` and the ``[a, b)`` bounds of ``window``
        (emulator-only; not CR).

        Only arguments that are not ``None`` are updated. Values apply to subsequent
        ``ACTIVATE`` instructions executed on this state.
        """
        if elu_alpha is not None:
            self.elu_alpha = float(elu_alpha)
        if window_a is not None:
            self.window_a = float(window_a)
        if window_b is not None:
            self.window_b = float(window_b)

    # -- register file snapshot (for VLIW dispatch) -------------------------

    def snapshot_regfile(self) -> RegFile:
        """Deep-copy the register file for VLIW read-before-write semantics."""
        return self.regfile.snapshot()

    # -- XMEM ↔ register transfers (mirrors ipu__load_r_reg / ipu__store_r_reg) --

    def write_constant_ones(self, xmem_addr: int) -> None:
        """Write a dtype-correct LANES-element ONES row with alias provenance.

        Use only for intentional identity constants, never ordinary inputs or
        weights. Any subsequent XMEM write overlapping this row revokes it.
        """
        lanes = _ipu().LANES
        if self.wide_vector_debug:
            fmt = "<f" if self.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<i"
            data = struct.pack(fmt, 1) * lanes
        else:
            from ipu_emu.ipu_math import dtype_one_byte
            data = bytes([dtype_one_byte(self.dtype)]) * lanes
        self.xmem.write_address(xmem_addr, data)
        self.xmem.mark_constant_ones(xmem_addr, data)

    # Debug/test conveniences, not on any instruction execution path. Each
    # takes a ROW number and mirrors its instruction in the active mode:
    # LDR_MULT_REG, LDR_CYCLIC_MULT_REG, STR_ACC_REG, LDR_MULT_MASK_REG.

    def _row_address(self, xmem_row: int) -> int:
        return xmem_row * _ipu().xmem_row_size_bytes(self)

    def load_r_reg_from_xmem(self, xmem_row: int, r_index: int) -> None:
        """Load one row from XMEM into R register *r_index* (0=R0, 1=R1)."""
        addr = self._row_address(xmem_row)
        if self.wide_vector_debug:
            data = self.xmem.read_address(addr, _ipu().xmem_row_size_bytes(self))
            self.regfile.set_r_wide_debug(r_index, data)
            name = "r_wide_debug"
        else:
            data = self.xmem.read_address(addr, _ipu().R_REG_SIZE)
            self.regfile.set_r(r_index, data)
            name = "r"
        if self.xmem.is_constant_ones(addr, data):
            self.regfile.mark_constant_ones(name, r_index * len(data), len(data))

    def store_r_reg_to_xmem(self, xmem_row: int, r_index: int) -> None:
        """Store R register *r_index* (one row) to XMEM."""
        data = (self.regfile.get_r_wide_debug(r_index) if self.wide_vector_debug
                else self.regfile.get_r(r_index))
        self.xmem.write_address(self._row_address(xmem_row), data)

    def load_r_cyclic_from_xmem(self, xmem_row: int, slot_element_idx: int = 0) -> None:
        """Load one row from XMEM into r_cyclic at *slot_element_idx*.

        An element index on a slot boundary (``R_CYCLIC_VALID_INDICES``), as
        for LDR_CYCLIC_MULT_REG. Defaults to slot 0.
        """
        ipu = _ipu()
        if slot_element_idx not in ipu.R_CYCLIC_VALID_INDICES:
            raise ValueError(f"slot_element_idx must be one of {ipu.R_CYCLIC_VALID_INDICES}; "
                             f"got {slot_element_idx}")
        addr = self._row_address(xmem_row)
        data = self.xmem.read_address(addr, ipu.xmem_row_size_bytes(self))
        byte_idx = slot_element_idx * ipu.xmem_element_width_bytes(self)
        name = "r_cyclic_wide_debug" if self.wide_vector_debug else "r_cyclic"
        getattr(self.regfile, f"set_{name}_at")(byte_idx, data)
        if self.xmem.is_constant_ones(addr, data):
            self.regfile.mark_constant_ones(name, byte_idx, len(data))

    def store_acc_to_xmem(self, xmem_row: int) -> None:
        """Store the whole accumulator (R_ACC_SIZE bytes in both modes) to XMEM."""
        self.xmem.write_address(self._row_address(xmem_row), self.regfile.get_r_acc_bytes())

    def load_r_mask_from_xmem(self, xmem_row: int) -> None:
        """Load the mask register (R_REG_SIZE bytes in both modes: 1 bit per lane)."""
        self.regfile.set_r_mask(
            self.xmem.read_address(self._row_address(xmem_row), _ipu().R_REG_SIZE))

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
