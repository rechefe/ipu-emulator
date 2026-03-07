"""IPU execution engine with automatic instruction dispatch.

This module contains the Ipu class which:
- Maintains IPU execution state (regfile, xmem, program counter, etc.)
- Implements all execute_* methods for each instruction
- Automatically dispatches instructions using instruction_spec metadata
- Extracts named operands from raw instruction dicts and passes them to handlers

Key Design:
- Single Ipu class contains all state and execution logic
- No manual opcode checking — uses instruction_spec.execute_fn for routing
- Each execute_* method receives NAMED operands (matching instruction_spec operand
  names), not raw instruction dicts. The dispatch layer parses the instruction and
  passes operand values as keyword arguments.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import Any

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.regfile import RegFile
from ipu_emu.ipu_math import ipu_mult, ipu_add, DType
from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    SLOT_BINARY_LAYOUT,
    SLOT_COUNT,
    get_instruction_by_opcode,
    create_emulator_constants,
)
from ipu_common.acc_stride_enums import (
    get_elements_per_row,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_common.registers import get_register_sizes, get_mult_stage_map

# ---------------------------------------------------------------------------
# Constants — derived from the single source of truth in ipu-common
# ---------------------------------------------------------------------------

_emu_constants = create_emulator_constants()
_reg_sizes = get_register_sizes()

# MultStageRegField: index → (register_name, element_index)
# e.g. [("r", 0), ("r", 1), ("mem_bypass", 0)]
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()

# Register dimensions — from REGISTER_DEFINITIONS via get_register_sizes()
LR_REG_COUNT = _reg_sizes["lr"]["count"]
R_REG_SIZE = _reg_sizes["r"]["size_bytes"]
R_CYCLIC_SIZE = _reg_sizes["r_cyclic"]["size_bytes"]
R_ACC_SIZE = _reg_sizes["r_acc"]["size_bytes"]


# ---------------------------------------------------------------------------
# Operand extraction: maps instruction_spec operand names → inst dict field keys
# ---------------------------------------------------------------------------

# Maps operand type string → field name suffix in the inst dict
# (derived from assembler token class names via camelCase→snake_case)
_TYPE_FIELD_SUFFIX = {
    "MultStageReg": "mult_stage_reg_field",
    "LrIdx": "lr_reg_field",
    "CrIdx": "cr_reg_field",
    "LcrIdx": "lcr_reg_field",
    "AaqRegIdx": "aaq_reg_field",
    "ElementsInRow": "elements_in_row_field",
    "HorizontalStride": "horizontal_stride_field",
    "VerticalStride": "vertical_stride_field",
    "Immediate": "lr_immediate_type",
    "BreakImmediate": "break_immediate_type",
    "Label": "label_token",
}

# Field prefix for each slot type (matches compound_inst naming)
_SLOT_FIELD_PREFIX = {
    "xmem": "xmem_inst",
    "mult": "mult_inst",
    "acc": "acc_inst",
    "cond": "cond_inst",
    "break": "break_inst",
    # LR is special: "lr_inst_0" and "lr_inst_1"
}


def _build_field_map_for_instruction(
    prefix: str, layout: list[str], operands: list[dict],
) -> dict[str, str]:
    """Build operand_name → inst dict field_key mapping for one instruction.

    Uses the same greedy matching algorithm as the assembler's
    _find_instruction_inst_mapping: for each operand, find the first
    unused position in the binary layout with matching type.
    """
    used = [False] * len(layout)
    field_map: dict[str, str] = {}

    for op in operands:
        op_type = op["type"]
        for j, layout_type in enumerate(layout):
            if layout_type == op_type and not used[j]:
                used[j] = True
                token_idx = j + 1  # +1 because token_0 is opcode
                suffix = _TYPE_FIELD_SUFFIX[layout_type]
                field_map[op["name"]] = f"{prefix}_token_{token_idx}_{suffix}"
                break

    return field_map


def _build_all_instruction_field_maps() -> dict[tuple, dict[str, str]]:
    """Precompute operand → field_key mappings for ALL instructions.

    Returns a dict keyed by:
      (slot_type, inst_name)       — for non-LR slots
      ("lr", inst_name, slot_idx)  — for LR sub-slots (0 and 1)
    """
    result: dict[tuple, dict[str, str]] = {}

    for slot_type, prefix in _SLOT_FIELD_PREFIX.items():
        layout = SLOT_BINARY_LAYOUT[slot_type]
        for inst_name, inst_def in INSTRUCTION_SPEC[slot_type].items():
            result[(slot_type, inst_name)] = _build_field_map_for_instruction(
                prefix, layout, inst_def["operands"]
            )

    # LR has multiple sub-slots with different field prefixes
    lr_layout = SLOT_BINARY_LAYOUT["lr"]
    for inst_name, inst_def in INSTRUCTION_SPEC["lr"].items():
        for slot_idx in range(SLOT_COUNT["lr"]):
            prefix = f"lr_inst_{slot_idx}"
            result[("lr", inst_name, slot_idx)] = _build_field_map_for_instruction(
                prefix, lr_layout, inst_def["operands"]
            )

    return result


# Precomputed at import time — no per-call overhead
_INSTRUCTION_FIELD_MAP = _build_all_instruction_field_maps()


def _build_read_operand_types() -> dict[tuple[str, str], dict[str, tuple[str, str]]]:
    """Precompute which operands need auto-resolution for each instruction.

    Returns dict keyed by (slot_type, inst_name) → {operand_name: (operand_type, source)}
    for operands with a ``"read"`` field (``"snapshot"`` or ``"live"``) in instruction_spec.
    """
    result: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    for slot_type, instructions in INSTRUCTION_SPEC.items():
        for inst_name, inst_def in instructions.items():
            read_ops = {
                op["name"]: (op["type"], op["read"])
                for op in inst_def["operands"]
                if "read" in op
            }
            if read_ops:
                result[(slot_type, inst_name)] = read_ops
    return result


_INSTRUCTION_READ_TYPES = _build_read_operand_types()


class BreakResult(IntEnum):
    CONTINUE = 0
    BREAK = 1


# ---------------------------------------------------------------------------
# IPU Execution Engine
# ---------------------------------------------------------------------------

class Ipu:
    """IPU execution engine with automatic instruction dispatch.

    This class encapsulates all IPU execution state and implements all
    instruction execution methods. Instructions are dispatched automatically
    based on instruction_spec metadata — no manual opcode checking needed.

    Each execute_* method receives named operand values (matching the operand
    names in INSTRUCTION_SPEC), not raw instruction dicts. The dispatch layer
    extracts operand values from the instruction word and passes them as
    keyword arguments.

    Attributes:
        state: IPU state (regfile, xmem, program counter, etc.)
        snapshot: Register file snapshot for VLIW read-before-write semantics
    """

    def __init__(self, state: IpuState):
        """Initialize IPU execution engine.

        Args:
            state: IPU state containing regfile, xmem, instruction memory, etc.
        """
        self.state = state
        self.snapshot: RegFile | None = None

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _resolve_operand(self, op_type: str, raw_value: int,
                         source: RegFile) -> int | bytearray:
        """Resolve a 'read' operand to its register value.

        Called by dispatch for operands with a ``"read"`` field in instruction_spec.
        The operand type determines how the raw index is resolved:

        - LrIdx → source.get_lr(idx) → uint32 value
        - CrIdx → source.get_cr(idx) → uint32 value
        - LcrIdx → LR if idx < LR_REG_COUNT, else CR → uint32 value
        - MultStageReg → register bytes via _MULT_STAGE_MAP → bytearray

        Args:
            op_type: Operand type string from instruction_spec.
            raw_value: Raw encoded index from the instruction word.
            source: Register file to read from (snapshot or live regfile).
        """
        if op_type == "LrIdx":
            return source.get_lr(raw_value)
        elif op_type == "CrIdx":
            return source.get_cr(raw_value)
        elif op_type == "LcrIdx":
            if raw_value < LR_REG_COUNT:
                return source.get_lr(raw_value)
            else:
                return source.get_cr(raw_value - LR_REG_COUNT)
        elif op_type == "MultStageReg":
            reg_name, elem_idx = _MULT_STAGE_MAP[raw_value]
            return source.get_register_bytes(reg_name, elem_idx)
        else:
            return raw_value

    def _mult_mask_and_shift(self, mask_idx: int, shift: int) -> None:
        """Apply mask-and-shift to mult_res, zeroing masked-out positions.

        Args:
            mask_idx: Mask slot selector value (already resolved from LR)
            shift: Shift amount value (already resolved from LR)
        """
        # Interpret shift as signed int32
        if shift >= 0x80000000:
            shift = shift - 0x100000000

        # The mask register is 128 bytes = 1024 bits.
        # It is accessed as an array of __uint128_t masks (8 masks of 128 bits each).
        # mask_idx selects which 128-bit mask to use.
        mask_bytes = self.state.regfile.get_r_mask()
        mask_slot = mask_idx % (R_REG_SIZE // 16)  # 128/16 = 8 slots of 128-bit
        # Extract 128-bit mask from the mask register
        offset = mask_slot * 16
        mask_int = int.from_bytes(mask_bytes[offset:offset + 16], byteorder="little")

        # Apply shift
        if shift > 0:
            mask_int <<= shift
        elif shift < 0:
            mask_int >>= -shift

        # Zero out mult_res where mask bit is set
        mult_res = self.state.regfile.raw("mult_res")
        for i in range(R_REG_SIZE):
            if (mask_int >> i) & 1:
                # Zero the uint32 word at position i
                struct.pack_into("<I", mult_res, i * 4, 0)

    # -----------------------------------------------------------------------
    # XMEM Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_xmem_nop(self) -> None:
        """Execute xmem_nop: No operation."""
        pass

    def execute_str_acc_reg(self, *, offset: int, base: int) -> None:
        """Execute str_acc_reg: Store accumulator to memory."""
        addr = offset + base
        acc_data = self.state.regfile.get_r_acc_bytes()
        self.state.xmem.write_address(addr, acc_data)

    def execute_ldr_mult_reg(self, *, dest: int, offset: int, base: int) -> None:
        """Execute ldr_mult_reg: Load data from memory into a mult stage register."""
        addr = offset + base
        data = self.state.xmem.read_address(addr, R_REG_SIZE)

        reg_name, elem_idx = _MULT_STAGE_MAP[dest]
        self.state.regfile.set_register_bytes(reg_name, elem_idx, data)

    def execute_ldr_cyclic_mult_reg(self, *, offset: int, base: int, index: int) -> None:
        """Execute ldr_cyclic_mult_reg: Load with cyclic addressing into r_cyclic."""
        addr = offset + base
        data = self.state.xmem.read_address(addr, R_REG_SIZE)

        assert index % R_REG_SIZE == 0, (
            f"LR index for cyclic load must be aligned to {R_REG_SIZE}: got {index}"
        )
        self.state.regfile.set_r_cyclic_at(index, data)

    def execute_ldr_mult_mask_reg(self, *, offset: int, base: int) -> None:
        """Execute ldr_mult_mask_reg: Load mask data from memory."""
        addr = offset + base
        data = self.state.xmem.read_address(addr, R_REG_SIZE)
        self.state.regfile.set_r_mask(data)

    # -----------------------------------------------------------------------
    # LR Instruction Handlers
    # -----------------------------------------------------------------------

    @staticmethod
    def _sign_extend_16(value: int) -> int:
        """Sign-extend a 16-bit value to a 32-bit signed integer."""
        if value & 0x8000:
            return value | 0xFFFF0000
        return value

    def execute_lr_incr(self, *, reg: int, value: int) -> None:
        """Execute incr: Increment a loop register by an immediate value."""
        current = self.state.regfile.get_lr(reg)
        signed_value = self._sign_extend_16(value)
        self.state.regfile.set_lr(reg, (current + signed_value) & 0xFFFFFFFF)

    def execute_lr_set(self, *, reg: int, value: int) -> None:
        """Execute set: Set a loop register to an immediate value."""
        self.state.regfile.set_lr(reg, self._sign_extend_16(value) & 0xFFFFFFFF)

    def execute_lr_add(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute add: Add two LCR registers."""
        self.state.regfile.set_lr(dest, (src_a + src_b) & 0xFFFFFFFF)

    def execute_lr_sub(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute sub: Subtract two LCR registers."""
        self.state.regfile.set_lr(dest, (src_a - src_b) & 0xFFFFFFFF)

    def _dispatch_lr_slots(self, inst: dict[str, int]) -> None:
        """Dispatch both LR sub-slots with conflict detection.

        LR is special: the VLIW word contains TWO LR sub-instructions
        (lr_inst_0 and lr_inst_1). Each is dispatched independently
        with named operands. Read operands are auto-resolved to values.
        """
        pending: list[tuple[str, str, dict[str, int]]] = []

        for slot_idx in range(SLOT_COUNT["lr"]):
            prefix = f"lr_inst_{slot_idx}"
            opcode_field = f"{prefix}_token_0_lr_inst_opcode"
            opcode = inst[opcode_field]

            inst_name, spec = get_instruction_by_opcode("lr", opcode)
            field_map = _INSTRUCTION_FIELD_MAP[("lr", inst_name, slot_idx)]
            kwargs = {name: inst[field_key] for name, field_key in field_map.items()}

            # incr by 0 is a NOP — skip
            if inst_name == "incr" and kwargs.get("value", 0) == 0:
                continue

            # Auto-resolve 'read' operands to register values.
            read_types = _INSTRUCTION_READ_TYPES.get(("lr", inst_name), {})
            for name, (op_type, source) in read_types.items():
                regfile = self.snapshot if source == "snapshot" else self.state.regfile
                kwargs[name] = self._resolve_operand(op_type, kwargs[name], regfile)

            pending.append((inst_name, spec["execute_fn"], kwargs))

        # Conflict check: no two valid instructions may write to the same LR
        lr_targets = [kw.get("reg", kw.get("dest")) for _, _, kw in pending]
        if len(lr_targets) != len(set(lr_targets)):
            raise RuntimeError(
                f"LR conflict: multiple writes to LR{lr_targets} in same cycle"
            )

        for _, fn_name, kwargs in pending:
            method = getattr(self, fn_name)
            method(**kwargs)

    # -----------------------------------------------------------------------
    # MULT Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_mult_nop(self) -> None:
        """Execute mult_nop: No operation."""
        pass

    def execute_mult_ee(self, *, ra: bytearray, cyclic_offset: int,
                        mask_offset: int, mask_shift: int) -> None:
        """Execute mult_ee: Element-wise multiplication."""
        dtype = self.state.get_cr_dtype()
        rb = self.state.regfile.get_r_cyclic_at(cyclic_offset, R_REG_SIZE)
        mult_res = self.state.regfile.raw("mult_res")

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ev(self, *, ra: bytearray, fixed_cyclic_idx: int,
                        mask_offset: int, mask_shift: int) -> None:
        """Execute mult_ev: Element x fixed cyclic multiplication."""
        dtype = self.state.get_cr_dtype()
        rb = self.state.regfile.get_r_cyclic_at(fixed_cyclic_idx, R_REG_SIZE)
        mult_res = self.state.regfile.raw("mult_res")

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[0], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ve(self, *, ra: bytearray, cyclic_offset: int,
                        mask_offset: int, mask_shift: int, fixed_ra_idx: int) -> None:
        """Execute mult_ve: Fixed Ra x cyclic element multiplication."""
        dtype = self.state.get_cr_dtype()
        rb = self.state.regfile.get_r_cyclic_at(cyclic_offset, R_REG_SIZE)
        mult_res = self.state.regfile.raw("mult_res")

        ra_fixed = ra[fixed_ra_idx % R_REG_SIZE]

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra_fixed, rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    # -----------------------------------------------------------------------
    # ACC Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_acc_nop(self) -> None:
        """Execute acc_nop: No operation."""
        pass

    def execute_reset_acc(self) -> None:
        """Execute reset_acc: Reset accumulator to zero."""
        self.state.regfile.set_r_acc_bytes(bytearray(R_ACC_SIZE))

    def execute_acc(self) -> None:
        """Execute acc: Accumulate mult_res into accumulator."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self.snapshot.raw("r_acc")
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = ipu_add(acc_val, mult_val, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_first(self) -> None:
        """Execute acc.first: Set r_acc to multiply result (no previous sum)."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            struct.pack_into(fmt, acc_buf, i * 4, mult_val)

    def execute_acc_add_aaq(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.add_aaq: Accumulate mult_res, then add aaq[aaq_rf_idx] to each of the 128 accumulator words."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self.snapshot.raw("r_acc")
        aaq_val = self.state.regfile.get_aaq(aaq_rf_idx)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = ipu_add(ipu_add(acc_val, mult_val, dtype), aaq_val, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_add_aaq_first(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.add_aaq.first: Set r_acc to mult_res + aaq[aaq_rf_idx] (no previous sum)."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        aaq_val = self.state.regfile.get_aaq(aaq_rf_idx)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = ipu_add(mult_val, aaq_val, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_stride(
        self,
        *,
        elements_in_row: int,
        horizontal_stride: int,
        vertical_stride: int,
        offset: int,
    ) -> None:
        """Execute acc.stride: Decimate mult_res by horizontal/vertical stride and write into r_acc.

        Operand semantics from ipu_common.acc_stride_enums (single source of truth).
        offset: LR value; (offset % 4) * 32 is the start index in r_acc (0, 32, 64, or 96).
        """
        elements_per_row = get_elements_per_row(elements_in_row)
        num_rows = R_REG_SIZE // elements_per_row

        h_enabled, h_inverted, h_expand = get_horizontal_stride_bits(horizontal_stride)
        v_enabled, v_inverted = get_vertical_stride_bits(vertical_stride)

        # Build list of source indices (0..127) or -1 for zero padding
        after_h: list[int] = []
        if not h_enabled:
            after_h = list(range(R_REG_SIZE))
            effective_row_len = elements_per_row
        else:
            half = elements_per_row // 2
            for row in range(num_rows):
                base = row * elements_per_row
                if h_inverted:
                    indices_in_row = [base + 1 + 2 * j for j in range(half)]
                else:
                    indices_in_row = [base + 2 * j for j in range(half)]
                if h_expand:
                    after_h.extend(indices_in_row)
                    after_h.extend([-1] * (elements_per_row - half))
                else:
                    after_h.extend(indices_in_row)
            effective_row_len = elements_per_row if h_expand else half

        num_rows_after_h = len(after_h) // effective_row_len
        if not v_enabled:
            out_indices = after_h
        else:
            out_indices = []
            row_sel = range(1, num_rows_after_h, 2) if v_inverted else range(0, num_rows_after_h, 2)
            for r in row_sel:
                start = r * effective_row_len
                out_indices.extend(after_h[start : start + effective_row_len])

        base = (offset % 4) * 32
        dtype = self.state.get_cr_dtype()
        fmt = "<i" if dtype == DType.INT8 else "<f"
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")

        for i, idx in enumerate(out_indices):
            if idx >= 0:
                val = struct.unpack_from(fmt, mult_res, idx * 4)[0]
            else:
                val = 0
            struct.pack_into(fmt, acc_buf, (base + i) * 4, val)

    # -----------------------------------------------------------------------
    # COND Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_beq(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute beq: Branch if equal."""
        self.state.program_counter = label if reg1 == reg2 else self.state.program_counter + 1

    def execute_bne(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute bne: Branch if not equal."""
        self.state.program_counter = label if reg1 != reg2 else self.state.program_counter + 1

    @staticmethod
    def _to_signed_32(value: int) -> int:
        """Interpret an unsigned 32-bit value as signed."""
        if value >= 0x80000000:
            return value - 0x100000000
        return value

    def execute_blt(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute blt: Branch if less than (signed comparison)."""
        s1 = self._to_signed_32(reg1)
        s2 = self._to_signed_32(reg2)
        self.state.program_counter = label if s1 < s2 else self.state.program_counter + 1

    def execute_bnz(self, *, test_reg: int, base_reg: int, label: int) -> None:
        """Execute bnz: Branch if not zero."""
        self.state.program_counter = label if test_reg != 0 else self.state.program_counter + 1

    def execute_bz(self, *, test_reg: int, base_reg: int, label: int) -> None:
        """Execute bz: Branch if zero."""
        self.state.program_counter = label if test_reg == 0 else self.state.program_counter + 1

    def execute_b(self, *, label: int) -> None:
        """Execute b: Unconditional branch."""
        self.state.program_counter = label

    def execute_br(self, *, reg: int) -> None:
        """Execute br: Branch to register value."""
        self.state.program_counter = reg

    def execute_bkpt(self) -> None:
        """Execute bkpt: Breakpoint (halt execution)."""
        self.state.program_counter = INST_MEM_SIZE  # halt

    # -----------------------------------------------------------------------
    # BREAK Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_break_nop(self) -> BreakResult:
        """Execute break_nop: No operation."""
        return BreakResult.CONTINUE

    def execute_break(self) -> BreakResult:
        """Execute break: Unconditional break."""
        return BreakResult.BREAK

    def execute_break_ifeq(self, *, reg: int, value: int) -> BreakResult:
        """Execute break_ifeq: Break if LR register equals immediate."""
        if reg == value:
            return BreakResult.BREAK
        return BreakResult.CONTINUE

    # -----------------------------------------------------------------------
    # Automatic Dispatch
    # -----------------------------------------------------------------------

    def dispatch_instruction(self, slot_type: str, inst: dict[str, int]) -> Any:
        """Dispatch instruction to the correct execute_* method with named operands.

        1. Reads the opcode from the instruction word
        2. Looks up the instruction spec (operand names, execute_fn)
        3. Extracts operand values from inst using precomputed field mappings
        4. Auto-resolves 'read' operands to register values
        5. Calls the handler with named keyword arguments

        Args:
            slot_type: Slot type ("xmem", "mult", "acc", "cond", "break")
            inst: Decoded instruction dict (field_name → int value)

        Returns:
            Return value from handler (BreakResult for break slot, None otherwise)
        """
        prefix = _SLOT_FIELD_PREFIX[slot_type]
        opcode_field = f"{prefix}_token_0_{slot_type}_inst_opcode"
        opcode = inst[opcode_field]

        # Look up instruction name and spec
        instruction_name, spec = get_instruction_by_opcode(slot_type, opcode)
        execute_fn_name = spec["execute_fn"]

        # Extract named operand values
        field_map = _INSTRUCTION_FIELD_MAP[(slot_type, instruction_name)]
        kwargs = {name: inst[field_key] for name, field_key in field_map.items()}

        # Auto-resolve 'read' operands to register values.
        read_types = _INSTRUCTION_READ_TYPES.get((slot_type, instruction_name), {})
        for name, (op_type, source) in read_types.items():
            regfile = self.snapshot if source == "snapshot" else self.state.regfile
            kwargs[name] = self._resolve_operand(op_type, kwargs[name], regfile)

        # Call handler with named arguments
        method = getattr(self, execute_fn_name)
        return method(**kwargs)

    # -----------------------------------------------------------------------
    # VLIW Execution
    # -----------------------------------------------------------------------

    def execute_vliw_cycle(self) -> BreakResult:
        """Execute one VLIW cycle.

        1. Fetch instruction at program counter
        2. Snapshot the register file
        3. Execute BREAK first (before side effects)
        4. Execute XMEM, LR, MULT, ACC, COND in parallel from the snapshot

        Returns:
            BreakResult.BREAK if break condition occurred, CONTINUE otherwise
        """
        inst = self.state.inst_mem[self.state.program_counter]
        if inst is None:
            # NOP — just advance PC
            self.state.program_counter += 1
            return BreakResult.CONTINUE

        self.snapshot = self.state.regfile.snapshot()

        # Break runs first — may halt before side effects
        result = self.dispatch_instruction("break", inst)
        if result == BreakResult.BREAK:
            return BreakResult.BREAK

        # Execute all other slots using the snapshot
        self._dispatch_lr_slots(inst)  # LR has dual sub-slots
        self.dispatch_instruction("xmem", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("cond", inst)

        return BreakResult.CONTINUE

    def execute_vliw_cycle_skip_break(self) -> None:
        """Execute the current instruction without re-checking break.

        Used after returning from a debug break to complete the cycle.
        """
        inst = self.state.inst_mem[self.state.program_counter]
        if inst is None:
            self.state.program_counter += 1
            return

        self.snapshot = self.state.regfile.snapshot()

        # Execute all slots except break
        self._dispatch_lr_slots(inst)
        self.dispatch_instruction("xmem", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("cond", inst)
