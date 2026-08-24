"""Top-level IPU state container.

Combines the register file, external memory, program counter, and instruction
memory into a single object — the Python equivalent of ``ipu__obj_t``.
"""

from __future__ import annotations

import struct
from enum import Enum
from typing import Any

from ipu_emu.regfile import RegFile
from ipu_emu.stats import RunStats
from ipu_emu.xmem import XMem
from ipu_common import activations as _activations
from ipu_common.registers import get_register_sizes
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
        xmem:            External memory (8 MB, mode-independent allocation).
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

        # --- Activation α (emulator-only; not mapped to CR) ----------------------
        self.elu_alpha: float = (
            float(elu_alpha) if elu_alpha is not None else float(_activations._ELU_ALPHA)
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
    ) -> None:
        """Write CR15 as dstructure configuration fields."""
        self.regfile.set_cr(
            CR_DSTRUCTURE_REG_INDEX,
            encode_dstructure(valid_elements=valid_elements, partition=partition, pad_mode=pad_mode),
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

    # -- mode-derived sizing (mirrors Ipu._element_width_bytes / _row_size_bytes /
    #    _r_cyclic_wrap_bytes in ipu.py -- duplicated here, not imported, because
    #    Ipu imports IpuState and importing back would be circular. Both read only
    #    wide_vector_debug/wide_vector_arithmetic, so keeping the two in sync is a
    #    one-line change on each side.) ------------------------------------------

    # r_acc is 128 uint32 lanes regardless of mode -- same derivation ipu.py's
    # LANES uses (R_ACC_SIZE // 4), read here directly from registers.py since
    # importing ipu.py's LANES would be circular (see note above).
    LANES = get_register_sizes()["r_acc"]["size_bytes"] // 4

    def _element_width_bytes(self) -> int:
        """Bytes per element in the active mode: 1 narrow, 4 wide-vector debug."""
        return 4 if self.wide_vector_debug else 1

    def _row_size_bytes(self) -> int:
        """Bytes per row (LANES elements) in the active mode: 128 narrow, 512 debug."""
        return self.LANES * self._element_width_bytes()

    def _r_cyclic_wrap_bytes(self) -> int:
        """r_cyclic's wrap modulus in the active mode: 512 B narrow, 2048 B debug."""
        return 512 * self._element_width_bytes()

    # -- XMEM ↔ register transfers (mirrors ipu__load_r_reg / ipu__store_r_reg) --
    #
    # Debug/test convenience helpers, not on any instruction execution path.
    # xmem_addr here is a ROW number, matching the corresponding instruction
    # (LDR_MULT_REG / STR_ACC_REG / etc. -- see #179); byte sizes are derived
    # from the active mode via the helpers above, not hardcoded.

    def load_r_reg_from_xmem(self, xmem_row: int, r_index: int) -> None:
        """Load one row from XMEM into R register *r_index* (0=R0, 1=R1).

        R0/R1 are fixed 128-byte registers in BOTH modes -- unlike r_cyclic,
        their allocated size does not scale with element width, because the
        real emulator represents debug-mode R0/R1 data as 128 Python
        floats/ints in ``_debug_mult_stage_vectors`` (the same staging LDR_MULT_REG
        uses), not as raw bytes in the register file. In debug mode this helper
        mirrors that: it unpacks the row into that staging list instead of
        writing register bytes.
        """
        data = self.xmem.read_address(xmem_row * self._row_size_bytes(), self._row_size_bytes())
        if self.wide_vector_debug:
            fmt = "<128f" if self.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<128i"
            self._debug_mult_stage_vectors[r_index] = list(struct.unpack_from(fmt, data, 0))
        else:
            self.regfile.set_r(r_index, data)

    def store_r_reg_to_xmem(self, xmem_row: int, r_index: int) -> None:
        """Store R register *r_index* (0=R0, 1=R1, one row) to XMEM.

        See load_r_reg_from_xmem: in debug mode this reads from
        ``_debug_mult_stage_vectors`` (the real backing for R0/R1 in that
        mode), not the 128-byte register file storage.
        """
        if self.wide_vector_debug:
            values = self._debug_mult_stage_vectors.get(
                r_index, [0.0 if self.wide_vector_arithmetic == WideVectorArithmetic.FP32 else 0] * self.LANES
            )
            fmt = "<128f" if self.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<128i"
            data = struct.pack(fmt, *values)
        else:
            data = self.regfile.get_r(r_index)
        self.xmem.write_address(xmem_row * self._row_size_bytes(), data)

    def load_r_cyclic_from_xmem(self, xmem_row: int, slot_element_idx: int = 0) -> None:
        """Load one row from XMEM into the cyclic register at *slot_element_idx*
        (an element index; must be one of the four slot boundaries 0/128/256/384,
        same as LDR_CYCLIC_MULT_REG -- see #180). Defaults to slot 0."""
        data = self.xmem.read_address(xmem_row * self._row_size_bytes(), self._row_size_bytes())
        byte_idx = slot_element_idx * self._element_width_bytes()
        self.regfile.set_r_cyclic_at(byte_idx, data, self._r_cyclic_wrap_bytes())

    def store_acc_to_xmem(self, xmem_row: int) -> None:
        """Store the accumulator (512 bytes, both modes -- width does not scale
        with element width) to XMEM."""
        data = self.regfile.get_r_acc_bytes()
        self.xmem.write_address(xmem_row * self._row_size_bytes(), data)

    def load_r_mask_from_xmem(self, xmem_row: int) -> None:
        """Load the mask register (128 bytes, both modes -- 1 bit/lane, does not
        scale with element width) from XMEM."""
        data = self.xmem.read_address(xmem_row * self._row_size_bytes(), 128)
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
