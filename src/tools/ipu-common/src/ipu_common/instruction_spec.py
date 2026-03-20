"""Master instruction specification for IPU — single source of truth.

This module defines ALL IPU instructions in one place with complete metadata:
- Instruction name and operands
- Operand definitions (name, type, meaning)
- Documentation (syntax, examples)
- Execution handler reference (for emulator)

KEY DESIGN PRINCIPLES:
1. NO EXPLICIT OPCODES: Opcode is derived from instruction position within slot.
   Position 0 → opcode 0, position 1 → opcode 1, etc.
   
2. NO CODE GENERATION: All opcode classes and constants are created at runtime
   using factory functions (create_assembler_opcodes, create_emulator_constants).
   
3. STRUCTURED OPERANDS: Each operand has a meaningful name and type.
   Types are string names resolved to actual classes by ipu_as.
   Operands with a ``"read"`` field are source registers whose values
   are auto-resolved by the emulator dispatcher before calling handlers.
   The value controls which register file is used:
     - ``"snapshot"`` → read from the VLIW snapshot (pre-write state)
     - ``"live"``     → read from the current (post-write) register file

OPERAND TYPE NAMES (resolved by ipu_as into actual token classes):
  - "MultStageReg": r0, r1, or mem_bypass (MultStageRegField)
  - "LrIdx": lr0-lr15 (LrRegField)  
  - "CrIdx": cr0-cr15 (CrRegField)
  - "LcrIdx": lr0-lr15 or cr0-cr15 (LcrRegField)
  - "Immediate": 32-bit signed integer (LrImmediateType)
  - "BreakImmediate": 16-bit break condition value (BreakImmediateType)
  - "Label": Branch target label (LabelToken)

SINGLE SOURCE OF TRUTH:
- Add an instruction → automatically gets opcode from position
- Rename instruction → assembler and emulator both see new name
- Change operands → both packages reflect the change
- No duplication, no manual opcode management

Structure:
    INSTRUCTION_SPEC = {
        "slot_type": {
            "instruction_name": {
                "operands": [
                    {"name": "src", "type": "LrIdx", "read": "snapshot"},
                    {"name": "dest", "type": "LrIdx"},
                    ...
                ],
                "doc": InstructionDoc(...),
                "execute_fn": "execute_my_instruction",
            },
            ...
        },
        ...
    }

    Operand "read" flag:
        - ``"read": "snapshot"`` → source register resolved from the VLIW
          snapshot (read-before-write). The emulator dispatcher reads the
          register value from the snapshot captured at cycle start.
        - ``"read": "live"`` → source register resolved from the current
          register file (sees writes from earlier slots in the same cycle).
        - Absent → destination/index operand; the raw index is passed
          through so the handler can write to it.
        - Immediates and Labels have no "read" flag (literal values always
          passed as-is).
        create_assembler_opcodes,
        create_emulator_constants,
    )
    
    # Get all instruction definitions
    spec = INSTRUCTION_SPEC
    
    # Extract opcodes automatically (derived from position)
    opcodes = extract_opcodes(spec)
    # → {"xmem": ["str_acc_reg", "ldr_mult_reg", ...], ...}
    
    # Create Opcode classes at runtime (no file generation)
    assembler_opcodes = create_assembler_opcodes()
    # → {"XmemInstOpcode": <class>, "LrInstOpcode": <class>, ...}
    
    # Create emulator constants at runtime
    constants = create_emulator_constants()
    # → {"XMEM_OP_STR_ACC_REG": 0, "XMEM_OP_LDR_MULT_REG": 1, ...}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Type


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "InstructionDoc",
    "INSTRUCTION_SPEC",
    "SLOT_BINARY_LAYOUT",
    "SLOT_COUNT",
    "extract_opcodes",
    "get_instruction",
    "get_instruction_by_opcode",
    "create_assembler_opcodes",
    "create_emulator_constants",
    "get_operand_names_and_types",
    "validate_instruction_spec",
]


# ===========================================================================
# Documentation Type
# ===========================================================================

@dataclass
class InstructionDoc:
    """Documentation for an instruction (used by assembler for help/docs)."""
    title: str
    summary: str
    syntax: str
    operands: list[str]
    operation: str | None = None
    example: str | None = None


# ===========================================================================
# SLOT BINARY LAYOUT
# ===========================================================================
# Defines the operand type positions in the binary encoding for each slot.
# Position 0 is always the opcode (implicit), these list the operand fields.
# This is the "union" of all operand positions a slot can encode.
#
# Example: XMEM can encode [MultStageReg, LrIdx, LrIdx, CrIdx] in its
# binary format. Individual instructions use a subset of these positions.
# ===========================================================================

SLOT_BINARY_LAYOUT: dict[str, list[str]] = {
    "xmem": ["MultStageReg", "LrIdx", "LrIdx", "CrIdx"],
    "mult": ["MultStageReg", "LrIdx", "LrIdx", "LrIdx", "LrIdx", "CrIdx", "AaqRegIdx"],
    "acc": ["AaqRegIdx", "ElementsInRow", "HorizontalStride", "VerticalStride", "LrIdx"],
    "aaq": ["AggMode", "PostFn", "CrIdx", "AaqRegIdx"],
    "lr": ["LrIdx", "LcrIdx", "LcrIdx", "Immediate"],
    "cond": ["LrIdx", "LrIdx", "Label"],
    "break": ["LrIdx", "BreakImmediate"],
}

# How many times each slot appears in the VLIW instruction word.
# Most slots appear once; LR appears twice (two independent sub-instructions).
SLOT_COUNT: dict[str, int] = {
    "break": 1,
    "xmem": 1,
    "mult": 1,
    "acc": 1,
    "aaq": 1,
    "lr": 2,
    "cond": 1,
}


# ===========================================================================
# MASTER INSTRUCTION SPECIFICATION
# ===========================================================================
# Each slot type (xmem, lr, mult, acc, cond, break) is defined separately.
# Instructions maintain ORDER — position in dict determines opcode!
# ===========================================================================

INSTRUCTION_SPEC = {
    # =========================================================================
    # XMEM Slot (Memory Load/Store Instructions)
    # Opcode = position in list: str_acc_reg=0, ldr_mult_reg=1, etc.
    # =========================================================================
    "xmem": {
        "str_acc_reg": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Store Accumulator",
                summary="Store accumulator to memory.",
                syntax="str_acc_reg offset base",
                operands=[
                    "offset: Offset register (lr0-lr15)",
                    "base: Base address register (cr0-cr15)",
                ],
                operation="Memory[offset + base] = r_acc",
                example="str_acc_reg cr0 cr1;;",
            ),
            "execute_fn": "execute_str_acc_reg",
        },
        "ldr_mult_reg": {
            "operands": [
                {"name": "dest", "type": "MultStageReg"},
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Register",
                summary="Load data from memory into a multiplication stage register.",
                syntax="ldr_mult_reg dest offset base",
                operands=[
                    "dest: Mult stage register (r0, r1, or mem_bypass)",
                    "offset: Offset register (lr0-lr15)",
                    "base: Base address register (cr0-cr15)",
                ],
                operation="dest = Memory[offset + base]",
                example="set lr0 0x1000;;\nldr_mult_reg r0 lr0 cr0;;",
            ),
            "execute_fn": "execute_ldr_mult_reg",
        },
        "ldr_cyclic_mult_reg": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
                {"name": "index", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Cyclic Register",
                summary="Load with cyclic addressing into r_cyclic.",
                syntax="ldr_cyclic_mult_reg offset base index",
                operands=[
                    "offset: Offset register (lr0-lr15)",
                    "base: Base address register (cr0-cr15)",
                    "index: Index inside cyclic register (lr0-lr15)",
                ],
                operation="r_cyclic[index % 512:128] = Memory[offset + base]",
            ),
            "execute_fn": "execute_ldr_cyclic_mult_reg",
        },
        "ldr_mult_mask_reg": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Mask Register",
                summary="Load mask data from memory.",
                syntax="ldr_mult_mask_reg offset base mask_idx",
                operands=[
                    "offset: Offset register (lr0-lr15)",
                    "base: Base address register (cr0-cr15)",
                ],
                operation="r_mask = Memory[offset + base]",
            ),
            "execute_fn": "execute_ldr_mult_mask_reg",
        },
        "xmem_nop": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (XMEM)",
                summary="No operation for xmem slot.",
                syntax="xmem_nop",
                operands=[],
            ),
            "execute_fn": "execute_xmem_nop",
        },
    },
    
    # =========================================================================
    # LR Slot (Loop Register Instructions)
    # Opcode = position: incr=0, set=1, add=2, sub=3
    # =========================================================================
    "lr": {
        "incr": {
            "operands": [
                {"name": "reg", "type": "LrIdx"},
                {"name": "value", "type": "Immediate"},
            ],
            "doc": InstructionDoc(
                title="Increment Loop Register",
                summary="Increment a loop register by an immediate value.",
                syntax="incr reg value",
                operands=[
                    "reg: Loop register to increment (lr0-lr15)",
                    "value: Immediate value to add",
                ],
                operation="reg += value",
                example="incr lr0 1;;",
            ),
            "execute_fn": "execute_lr_incr",
        },
        "set": {
            "operands": [
                {"name": "reg", "type": "LrIdx"},
                {"name": "value", "type": "Immediate"},
            ],
            "doc": InstructionDoc(
                title="Set Loop Register",
                summary="Set a loop register to an immediate value.",
                syntax="set reg value",
                operands=[
                    "reg: Loop register (lr0-lr15)",
                    "value: 32-bit immediate value",
                ],
                operation="reg = value",
                example="set lr0 0x1000;;",
            ),
            "execute_fn": "execute_lr_set",
        },
        "add": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "src_a", "type": "LcrIdx", "read": "snapshot"},
                {"name": "src_b", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Add Registers",
                summary="Add two registers and store in destination.",
                syntax="add dest src_a src_b",
                operands=[
                    "dest: Destination loop register (lr0-lr15)",
                    "src_a: First source register (lr0-lr15 or cr0-cr15)",
                    "src_b: Second source register (lr0-lr15 or cr0-cr15)",
                ],
                operation="dest = src_a + src_b",
                example="add lr0 lr1 lr2;;",
            ),
            "execute_fn": "execute_lr_add",
        },
        "sub": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "src_a", "type": "LcrIdx", "read": "snapshot"},
                {"name": "src_b", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Subtract Registers",
                summary="Subtract two registers and store in destination.",
                syntax="sub dest src_a src_b",
                operands=[
                    "dest: Destination loop register (lr0-lr15)",
                    "src_a: First source register (lr0-lr15 or cr0-cr15)",
                    "src_b: Second source register (lr0-lr15 or cr0-cr15)",
                ],
                operation="dest = src_a - src_b",
                example="sub lr0 lr1 lr2;;",
            ),
            "execute_fn": "execute_lr_sub",
        },
    },
    
    # =========================================================================
    # MULT Slot (Multiply Instructions)
    # Opcode = position: mult.ee=0, mult.ve=1, mult.ve.cr=2, mult.ve.aaq=3, mult_nop=4
    # =========================================================================
    "mult": {
        "mult.ee": {
            "operands": [
                {"name": "ra", "type": "MultStageReg", "read": "live"},
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Element-wise Multiply",
                summary="Multiply elements of two registers element by element.",
                syntax="mult.ee ra cyclic_offset mask_offset mask_shift",
                operands=[
                    "ra: Multiplicand register (r0, r1, or mem_bypass)",
                    "cyclic_offset: Base offset for multiplier from RC (cyclic register)",
                    "mask_offset: Offset to select mask from RM (mask register)",
                    "mask_shift: Shift applied to the mask register",
                ],
                operation="Element-wise multiply with masking",
                example="mult.ee r0 lr0 lr1 lr2;;",
            ),
            "execute_fn": "execute_mult_ee",
        },
        "mult.ve": {
            "operands": [
                {"name": "ra", "type": "MultStageReg", "read": "live"},
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "fixed_ra_idx", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Vector-Element Multiply",
                summary="Multiply a fixed element from Ra register against cyclic register elements.",
                syntax="mult.ve ra cyclic_offset mask_offset mask_shift fixed_ra_idx",
                operands=[
                    "ra: Multiplicand register (r0, r1, or mem_bypass)",
                    "cyclic_offset: Base offset for multiplier from RC (cyclic register)",
                    "mask_offset: Offset to select mask from RM (mask register)",
                    "mask_shift: Shift applied to the mask register",
                    "fixed_ra_idx: Fixed index for element selection from Ra register",
                ],
                operation=(
                    "For each i: result[i] = RA[fixed_ra_idx] * RC[cyclic_offset + i].\n"
                    "If cyclic_offset + i >= R_CYCLIC_SIZE, the cyclic element is replaced by\n"
                    "the dtype-specific constant 1 (int8: 1, f8e4m3: 0x38, f8e5m2: 0x3C)."
                ),
                example="mult.ve r0 lr0 lr1 lr2 lr3;;",
            ),
            "execute_fn": "execute_mult_ve",
        },
        "mult.ve.cr": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "cr_idx", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Vector-CR Multiply",
                summary="Multiply cyclic register elements against a CR register scalar.",
                syntax="mult.ve.cr cyclic_offset mask_offset mask_shift cr_idx",
                operands=[
                    "cyclic_offset: Base offset for multiplier from RC (cyclic register)",
                    "mask_offset: Offset to select mask from RM (mask register)",
                    "mask_shift: Shift applied to the mask register",
                    "cr_idx: CR register whose low byte is used as the scalar multiplicand",
                ],
                operation=(
                    "For each i: result[i] = (CR[cr_idx] & 0xFF) * RC[cyclic_offset + i].\n"
                    "If cyclic_offset + i >= R_CYCLIC_SIZE, the cyclic element is replaced by\n"
                    "the dtype-specific constant 1 (int8: 1, f8e4m3: 0x38, f8e5m2: 0x3C)."
                ),
                example="mult.ve.cr lr0 lr1 lr2 cr3;;",
            ),
            "execute_fn": "execute_mult_ve_cr",
        },
        "mult.ve.aaq": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Vector-AAQ Multiply",
                summary="Multiply cyclic register elements against an AAQ register scalar.",
                syntax="mult.ve.aaq cyclic_offset mask_offset mask_shift aaq_rf_idx",
                operands=[
                    "cyclic_offset: Base offset for multiplier from RC (cyclic register)",
                    "mask_offset: Offset to select mask from RM (mask register)",
                    "mask_shift: Shift applied to the mask register",
                    "aaq_rf_idx: AAQ register index whose low byte is used as the scalar multiplicand",
                ],
                operation=(
                    "For each i: result[i] = (AAQ[aaq_rf_idx] & 0xFF) * RC[cyclic_offset + i].\n"
                    "If cyclic_offset + i >= R_CYCLIC_SIZE, the cyclic element is replaced by\n"
                    "the dtype-specific constant 1 (int8: 1, f8e4m3: 0x38, f8e5m2: 0x3C)."
                ),
                example="mult.ve.aaq lr0 lr1 lr2 aaq0;;",
            ),
            "execute_fn": "execute_mult_ve_aaq",
        },
        "mult_nop": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (MULT)",
                summary="No operation for multiply slot.",
                syntax="mult_nop",
                operands=[],
            ),
            "execute_fn": "execute_mult_nop",
        },
    },

    # =========================================================================
    # ACC Slot (Accumulator Instructions)
    # Opcode = position: acc=0, acc.first=1, reset_acc=2, acc_nop=3, acc.add_aaq=4, acc.add_aaq.first=5, acc.max=6, acc.max.first=7, acc.stride=8
    # =========================================================================
    "acc": {
        "acc": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate",
                summary="Accumulate multiply result.",
                syntax="acc",
                operands=[],
                operation="r_acc += multiply_result",
            ),
            "execute_fn": "execute_acc",
        },
        "acc.first": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate First",
                summary="Set accumulator to multiply result (do not add to previous r_acc).",
                syntax="acc.first",
                operands=[],
                operation="r_acc = multiply_result",
                example="acc.first;;",
            ),
            "execute_fn": "execute_acc_first",
        },
        "reset_acc": {
            "operands": [],
            "doc": InstructionDoc(
                title="Reset Accumulator",
                summary="Reset accumulator to zero.",
                syntax="reset_acc",
                operands=[],
                operation="r_acc = 0",
            ),
            "execute_fn": "execute_reset_acc",
        },
        "acc_nop": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (ACC)",
                summary="No operation for accumulator slot.",
                syntax="acc_nop",
                operands=[],
            ),
            "execute_fn": "execute_acc_nop",
        },
        "acc.add_aaq": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulate and Add AAQ",
                summary="Accumulate multiply result, then add the selected AAQ register (32-bit) to each of the 128 accumulator words.",
                syntax="acc.add_aaq aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (aaq0-aaq3)",
                ],
                operation=(
                    "r_acc += multiply_result;\n"
                    "for i in [0, 128): r_acc[i] += aaq_regs[aaq_rf_idx]"
                ),
                example="acc.add_aaq aaq0;;",
            ),
            "execute_fn": "execute_acc_add_aaq",
        },
        "acc.add_aaq.first": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulate and Add AAQ (First)",
                summary="Set accumulator to multiply result plus selected AAQ register (do not add to previous r_acc).",
                syntax="acc.add_aaq.first aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (aaq0-aaq3)",
                ],
                operation=(
                    "r_acc = multiply_result;\n"
                    "for i in [0, 128): r_acc[i] += aaq_regs[aaq_rf_idx]"
                ),
                example="acc.add_aaq.first aaq0;;",
            ),
            "execute_fn": "execute_acc_add_aaq_first",
        },
        "acc.max": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Max",
                summary="For each element, set r_acc[i] = max(r_acc[i], mult_res[i], aaq_reg[aaq_rf_idx]).",
                syntax="acc.max aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (aaq0-aaq3)",
                ],
                operation=(
                    "for i in [0, 128): r_acc[i] = max(r_acc[i], mult_res[i], aaq_regs[aaq_rf_idx])"
                ),
                example="acc.max aaq0;;",
            ),
            "execute_fn": "execute_acc_max",
        },
        "acc.max.first": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Max (First)",
                summary="For each element, set r_acc[i] = max(mult_res[i], aaq_reg[aaq_rf_idx]). Previous r_acc is ignored (treated as 0).",
                syntax="acc.max.first aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (aaq0-aaq3)",
                ],
                operation=(
                    "for i in [0, 128): r_acc[i] = max(mult_res[i], aaq_regs[aaq_rf_idx])"
                ),
                example="acc.max.first aaq0;;",
            ),
            "execute_fn": "execute_acc_max_first",
        },
        "acc.stride": {
            "operands": [
                {"name": "elements_in_row", "type": "ElementsInRow"},
                {"name": "horizontal_stride", "type": "HorizontalStride"},
                {"name": "vertical_stride", "type": "VerticalStride"},
                {"name": "offset", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Stride",
                summary="Reorder the multiplication result into r_acc using horizontal/vertical stride decimation. Only updates the RACC indexes written; leaves the rest unchanged.",
                syntax="acc.stride elements_in_row horizontal_stride vertical_stride offset",
                operands=[
                    "elements_in_row: Elements per row (8, 16, 32, or 64)",
                    "horizontal_stride: Horizontal stride mode (enabled, inverted, expand)",
                    "vertical_stride: Vertical stride mode (enabled, inverted)",
                    "offset: LR register; value % 4 gives start index in RACC (0, 32, 64, or 96)",
                ],
                operation=(
                    "Decimate mult_res as rows×cols; apply horizontal stride (take every 2nd column, optional expand); "
                    "then vertical stride (take every 2nd row). Write result into r_acc[start:start+N] where start = (offset%4)*32, N = 32|64|128."
                ),
                example="acc.stride 8 off off lr0;;",
            ),
            "execute_fn": "execute_acc_stride",
        },
    },

    # =========================================================================
    # AAQ Slot (Activation and Quantization)
    # Opcode = position: aaq_nop=0, agg=1
    # =========================================================================
    "aaq": {
        "aaq_nop": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (AAQ)",
                summary="No operation for AAQ slot.",
                syntax="aaq_nop",
                operands=[],
            ),
            "execute_fn": "execute_aaq_nop",
        },
        "agg": {
            "operands": [
                {"name": "agg_mode", "type": "AggMode"},
                {"name": "post_fn", "type": "PostFn"},
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Aggregate",
                summary="Collapse 128 r_acc words into one value (SUM or MAX), apply post function, store to selected AAQ register.",
                syntax="agg agg_mode post_fn cr_idx aaq_rf_idx",
                operands=[
                    "agg_mode: sum or max",
                    "post_fn: value, value_cr, inv, or inv_sqrt",
                    "cr_idx: CR register for value_cr post function (cr0-cr15)",
                    "aaq_rf_idx: AAQ register to store result (aaq0-aaq3)",
                ],
                operation=(
                    "If sum: v = sum(r_acc[0..127]). "
                    "If max: v = max(r_acc[0..127], aaq[aaq_rf_idx]). "
                    "Apply post_fn(v): value→v, value_cr→v*cr[cr_idx], inv→1/v, inv_sqrt→1/sqrt(v). "
                    "aaq[aaq_rf_idx] = result."
                ),
                example="agg sum value cr0 aaq0;;",
            ),
            "execute_fn": "execute_agg",
        },
    },

    # =========================================================================
    # COND Slot (Conditional Branch Instructions)
    # Opcode = position: beq=0, bne=1, blt=2, bnz=3, bz=4, b=5, br=6, bkpt=7
    # =========================================================================
    "cond": {
        "beq": {
            "operands": [
                {"name": "reg1", "type": "LrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Equal",
                summary="Branch if two registers are equal.",
                syntax="beq reg1 reg2 label",
                operands=[
                    "reg1: First register to compare (lr0-lr15)",
                    "reg2: Second register to compare (lr0-lr15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 == reg2) PC = label",
                example="beq lr0 lr1 end;;",
            ),
            "execute_fn": "execute_beq",
        },
        "bne": {
            "operands": [
                {"name": "reg1", "type": "LrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Not Equal",
                summary="Branch if two registers are not equal.",
                syntax="bne reg1 reg2 label",
                operands=[
                    "reg1: First register to compare (lr0-lr15)",
                    "reg2: Second register to compare (lr0-lr15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 != reg2) PC = label",
                example="bne lr0 lr1 different;;",
            ),
            "execute_fn": "execute_bne",
        },
        "blt": {
            "operands": [
                {"name": "reg1", "type": "LrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Less Than",
                summary="Branch if first register is less than second.",
                syntax="blt reg1 reg2 label",
                operands=[
                    "reg1: First register to compare (lr0-lr15)",
                    "reg2: Second register to compare (lr0-lr15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 < reg2) PC = label",
                example="blt lr0 lr1 smaller;;",
            ),
            "execute_fn": "execute_blt",
        },
        "bnz": {
            "operands": [
                {"name": "test_reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "base_reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Not Zero",
                summary="Branch if test register not equal to base register.",
                syntax="bnz test_reg base_reg label",
                operands=[
                    "test_reg: Register to test (lr0-lr15)",
                    "base_reg: Base comparison register (lr0-lr15)",
                    "label: Branch target label",
                ],
                operation="if (test_reg != base_reg) PC = label",
                example="bnz lr3 lr0 loop;;",
            ),
            "execute_fn": "execute_bnz",
        },
        "bz": {
            "operands": [
                {"name": "test_reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "base_reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Zero",
                summary="Branch if test register equals base register.",
                syntax="bz test_reg base_reg label",
                operands=[
                    "test_reg: Register to test (lr0-lr15)",
                    "base_reg: Base comparison register (lr0-lr15)",
                    "label: Branch target label",
                ],
                operation="if (test_reg == base_reg) PC = label",
                example="bz lr0 lr1 zero;;",
            ),
            "execute_fn": "execute_bz",
        },
        "b": {
            "operands": [
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Unconditional Branch",
                summary="Always branch to label.",
                syntax="b label",
                operands=["label: Branch target label"],
                operation="PC = label",
                example="b start;;",
            ),
            "execute_fn": "execute_b",
        },
        "br": {
            "operands": [
                {"name": "reg", "type": "LrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Branch Register",
                summary="Branch to address in register.",
                syntax="br reg",
                operands=["reg: Register containing target address (lr0-lr15)"],
                operation="PC = reg",
            ),
            "execute_fn": "execute_br",
        },
        "bkpt": {
            "operands": [],
            "doc": InstructionDoc(
                title="Breakpoint",
                summary="Conditional breakpoint.",
                syntax="bkpt",
                operands=[],
                operation="Halt execution (debugging)",
            ),
            "execute_fn": "execute_bkpt",
        },
    },

    # =========================================================================
    # BREAK Slot (Break Instructions)
    # Opcode = position: break=0, break.ifeq=1, break_nop=2
    # =========================================================================
    "break": {
        "break": {
            "operands": [],
            "doc": InstructionDoc(
                title="Break",
                summary="Unconditional break.",
                syntax="break",
                operands=[],
                operation="Halt execution",
            ),
            "execute_fn": "execute_break",
        },
        "break.ifeq": {
            "operands": [
                {"name": "reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "value", "type": "BreakImmediate"},
            ],
            "doc": InstructionDoc(
                title="Break if Equal",
                summary="Break execution if register equals value.",
                syntax="break.ifeq reg value",
                operands=[
                    "reg: Register to test (lr0-lr15)",
                    "value: Immediate value to compare against",
                ],
                operation="if (reg == value) BREAK",
                example="break.ifeq lr0 10;;",
            ),
            "execute_fn": "execute_break_ifeq",
        },
        "break_nop": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (BREAK)",
                summary="No operation for break slot.",
                syntax="break_nop",
                operands=[],
            ),
            "execute_fn": "execute_break_nop",
        },
    },
}


# ===========================================================================
# Query Functions (position-based opcode derivation)
# ===========================================================================

def get_opcode_for_instruction(slot_type: str, instruction_name: str) -> int:
    """Get opcode index for an instruction (derived from its position).
    
    Args:
        slot_type: Slot type (e.g., "xmem", "lr", "mult")
        instruction_name: Instruction name (e.g., "str_acc_reg")
    
    Returns:
        Opcode index (0-based position in slot's instruction list)
    
    Raises:
        KeyError if slot or instruction not found
    """
    instructions = INSTRUCTION_SPEC[slot_type]
    for idx, name in enumerate(instructions.keys()):
        if name == instruction_name:
            return idx
    raise KeyError(f"Instruction '{instruction_name}' not found in slot '{slot_type}'")


def extract_opcodes() -> Dict[str, List[str]]:
    """Extract all instruction names grouped by slot type (in order).
    
    Returns a dict mapping slot type to list of instruction names, where
    the position in the list IS the opcode index.
    
    Example:
        {
            "xmem": ["str_acc_reg", "ldr_mult_reg", ...],
            "lr": ["incr", "set", "add", "sub"],
            ...
        }
    """
    result = {}
    for slot_type, instructions in INSTRUCTION_SPEC.items():
        result[slot_type] = list(instructions.keys())
    return result


def get_instruction(slot_type: str, instruction_name: str) -> dict:
    """Look up a specific instruction definition.
    
    Args:
        slot_type: Slot type (e.g., "xmem", "lr", "mult")
        instruction_name: Instruction name (e.g., "str_acc_reg")
    
    Returns:
        The instruction definition dict with "operands", "doc", "execute_fn"
    
    Raises:
        KeyError if instruction not found
    """
    return INSTRUCTION_SPEC[slot_type][instruction_name]


def get_instruction_by_opcode(slot_type: str, opcode_index: int) -> Tuple[str, dict]:
    """Look up instruction by slot type and opcode (position-based).
    
    Args:
        slot_type: Slot type (e.g., "xmem", "lr")
        opcode_index: 0-based opcode index
    
    Returns:
        Tuple of (instruction_name, instruction_definition)
    
    Raises:
        KeyError if opcode index out of range
    """
    instructions = INSTRUCTION_SPEC[slot_type]
    instruction_name = list(instructions.keys())[opcode_index]
    return instruction_name, instructions[instruction_name]


def get_operand_names_and_types(slot_type: str, instruction_name: str) -> List[Tuple[str, str]]:
    """Get operand names and types for an instruction.
    
    Returns:
        List of (operand_name, operand_type) tuples
        
    Example:
        [("offset", "CrIdx"), ("base", "CrIdx")]
    """
    inst_def = get_instruction(slot_type, instruction_name)
    return [(op["name"], op["type"]) for op in inst_def["operands"]]


# ===========================================================================
# Runtime Opcode Factories (no code generation!)
# ===========================================================================

def create_assembler_opcodes() -> Dict[str, Type]:
    """Create Opcode classes at runtime (no code generation needed).
    
    Returns a dict of dynamically created Opcode subclasses:
        {
            "XmemInstOpcode": <class>,
            "LrInstOpcode": <class>,
            ...
        }
    
    Each class has an enum_array() classmethod returning instruction names
    in order (position = opcode).
    """
    import ipu_as.ipu_token as ipu_token
    
    class Opcode(ipu_token.EnumToken):
        """Base class for all opcode types."""
        @classmethod
        def find_opcode_class(cls, opcode_token) -> Type["Opcode"]:
            for subclass in cls.__subclasses__():
                if opcode_token.value in subclass.enum_array():
                    return subclass
            raise ValueError(
                f"Opcode '{opcode_token.value}' not found in any Opcode subclass"
            )
    
    slot_to_class_name = {
        "xmem": "XmemInstOpcode",
        "lr": "LrInstOpcode",
        "mult": "MultInstOpcode",
        "acc": "AccInstOpcode",
        "aaq": "AaqInstOpcode",
        "cond": "CondInstOpcode",
        "break": "BreakInstOpcode",
    }
    
    result = {}
    for slot_type, class_name in slot_to_class_name.items():
        instructions = INSTRUCTION_SPEC[slot_type]
        enum_array = list(instructions.keys())
        
        # Create class dynamically
        opcode_class = type(
            class_name,
            (Opcode,),
            {
                "enum_array": classmethod(lambda cls, ea=enum_array: ea),
            }
        )
        result[class_name] = opcode_class
    
    return result


def create_emulator_constants() -> Dict[str, int]:
    """Create emulator opcode constants at runtime.
    
    Returns a dict of constants like:
        {
            "XMEM_OP_STR_ACC_REG": 0,
            "XMEM_OP_LDR_MULT_REG": 1,
            ...
            "NUM_XMEM_OP": 5,
            ...
        }
    """
    slot_to_prefix = {
        "xmem": "XMEM_OP",
        "lr": "LR_OP",
        "mult": "MULT_OP",
        "acc": "ACC_OP",
        "aaq": "AAQ_OP",
        "cond": "COND_OP",
        "break": "BREAK_OP",
    }
    
    constants = {}
    
    for slot_type, instructions in INSTRUCTION_SPEC.items():
        prefix = slot_to_prefix[slot_type]
        
        # Add opcode constants for each instruction
        for opcode_idx, instruction_name in enumerate(instructions.keys()):
            # Convert name to constant style: "str_acc_reg" → "STR_ACC_REG"
            const_name = instruction_name.upper().replace(".", "_").replace("-", "_")
            constants[f"{prefix}_{const_name}"] = opcode_idx
        
        # Add count constant
        constants[f"NUM_{prefix}"] = len(instructions)
    
    return constants


# ===========================================================================
# Validation
# ===========================================================================

def validate_instruction_spec() -> None:
    """Validate instruction specification consistency.
    
    Checks:
    - All slot types exist
    - All instructions have required fields (operands, doc, execute_fn)
    - Operands have name and type fields
    - All operand types are recognized
    
    Raises ValueError if validation fails.
    """
    valid_operand_types = {
        "MultStageReg", "LrIdx", "CrIdx", "LcrIdx", "AaqRegIdx",
        "ElementsInRow", "HorizontalStride", "VerticalStride",
        "AggMode", "PostFn",
        "Immediate", "BreakImmediate", "Label"
    }
    valid_read_sources = {"snapshot", "live"}
    
    for slot_type, instructions in INSTRUCTION_SPEC.items():
        if not isinstance(instructions, dict):
            raise ValueError(f"Slot '{slot_type}' instructions must be a dict")
        
        if len(instructions) == 0:
            raise ValueError(f"Slot '{slot_type}' has no instructions")
        
        for inst_name, inst_def in instructions.items():
            # Check required fields
            if "operands" not in inst_def:
                raise ValueError(f"{slot_type}.{inst_name}: missing 'operands' field")
            if "doc" not in inst_def:
                raise ValueError(f"{slot_type}.{inst_name}: missing 'doc' field")
            if "execute_fn" not in inst_def:
                raise ValueError(f"{slot_type}.{inst_name}: missing 'execute_fn' field")
            
            # Validate operands structure
            operands = inst_def["operands"]
            if not isinstance(operands, list):
                raise ValueError(
                    f"{slot_type}.{inst_name}: 'operands' must be a list of dicts"
                )
            
            for operand in operands:
                if not isinstance(operand, dict):
                    raise ValueError(
                        f"{slot_type}.{inst_name}: each operand must be a dict"
                    )
                if "name" not in operand:
                    raise ValueError(
                        f"{slot_type}.{inst_name}: operand missing 'name' field"
                    )
                if "type" not in operand:
                    raise ValueError(
                        f"{slot_type}.{inst_name}: operand '{operand['name']}' "
                        f"missing 'type' field"
                    )
                
                op_type = operand["type"]
                if op_type not in valid_operand_types:
                    raise ValueError(
                        f"{slot_type}.{inst_name}: operand '{operand['name']}' "
                        f"has invalid type '{op_type}'. Must be one of: "
                        f"{valid_operand_types}"
                    )
                
                # Validate 'read' field if present
                if "read" in operand:
                    read_val = operand["read"]
                    if read_val not in valid_read_sources:
                        raise ValueError(
                            f"{slot_type}.{inst_name}: operand '{operand['name']}' "
                            f"has invalid 'read' value '{read_val}'. "
                            f"Must be one of: {valid_read_sources}"
                        )
            
            # Validate doc is InstructionDoc
            if not isinstance(inst_def["doc"], InstructionDoc):
                raise ValueError(
                    f"{slot_type}.{inst_name}: 'doc' must be InstructionDoc instance"
                )
            
            # Validate execute_fn is a string
            if not isinstance(inst_def["execute_fn"], str):
                raise ValueError(
                    f"{slot_type}.{inst_name}: 'execute_fn' must be a string"
                )


# Validate on import
validate_instruction_spec()
