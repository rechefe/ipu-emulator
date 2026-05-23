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
  - "MultStageReg": R0 or R1 (MultStageRegField); 2-bit encoding in the VLIW word
  - "LrIdx": LR0–LR15 (LrRegField)  
  - "CrIdx": CR0–CR15 (CrRegField)
  - "LcrIdx": LR0–LR15 or CR0–CR15 (LcrRegField)
  - "AddSubSrcB": second operand for ADD/SUB — LR, CR, or unsigned IMM5 (AddSubSrcBField; 6-bit encoding)
  - "LrModPow2KImmediate": k operand for INCR_MOD_POW2 (semantic k ∈ [1, 9]; encoded as k−1 in 4 bits)
  - "MultMaskOffsetImmediate": mask slot index for mult masking (0–7; eight 128-bit slots in R_MASK)
  - "ActivationFn": keyword on `ACTIVATE` (see ``ACTIVATION_FN_NAMES`` in ``activations.py``)
  - "BreakImmediate": 16-bit BREAK condition value (BreakImmediateType)
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
    "SLOT_UNIONS",
    "SLOT_COUNT",
    "VALID_OPERAND_TYPES",
    "SlotUnion",
    "canonical_instruction_name",
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
# SLOT BINARY LAYOUT  (derived — do not edit manually)
# ===========================================================================
# SLOT_BINARY_LAYOUT and SLOT_UNIONS are computed by the union layout solver
# from INSTRUCTION_SPEC below.  They are populated after INSTRUCTION_SPEC is
# fully defined (see the call at the bottom of this module).
#
# SLOT_BINARY_LAYOUT: slot → [canonical_type, ...] — one entry per union
#   field, in field order.  Used by the assembler's Inst.operand_types() and
#   the emulator's field-map builder.
#
# SLOT_UNIONS: slot → SlotUnion — full union metadata including per-opcode
#   operand-to-field bindings.
# ===========================================================================

# Placeholder — overwritten after INSTRUCTION_SPEC is defined.
SLOT_BINARY_LAYOUT: dict[str, list[str]] = {}
SLOT_UNIONS: dict = {}

# How many times each slot appears in the VLIW instruction word.
# Most slots appear once; LR appears three times (three independent sub-instructions).
SLOT_COUNT: dict[str, int] = {
    "break": 1,
    "xmem": 1,
    "mult": 1,
    "acc": 1,
    "aaq": 1,
    "lr": 3,
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
    # Opcode = position in list: STR_ACC_REG=0, LDR_MULT_REG=1, etc.
    # =========================================================================
    "xmem": {
        "STR_ACC_REG": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Store Accumulator",
                summary="Store accumulator to memory.",
                syntax="STR_ACC_REG offset, base",
                operands=[
                    "offset: Offset register (LR0–LR15)",
                    "base: Base address register (CR0–CR15)",
                ],
                operation="Memory[offset + base] = R_ACC",
                example="STR_ACC_REG CR0, CR1;;",
            ),
            "execute_fn": "execute_str_acc_reg",
        },
        "LDR_MULT_REG": {
            "operands": [
                {"name": "dest", "type": "MultStageReg"},
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Register",
                summary="Load data from memory into a multiplication stage register.",
                syntax="LDR_MULT_REG dest, offset, base",
                operands=[
                    "`dest`: **`R0`** | **`R1`** — mult-stage register to load (2-bit field; only these encodings are valid).",
                    "`offset`: **`LR0`**…**`LR15`** — offset register.",
                    "`base`: **`CR0`**…**`CR15`** — base address register.",
                ],
                operation="dest = Memory[offset + base]  # 128 elements (512 in wide-vector debug mode)",
                example="SET LR0, CR1;;\nLDR_MULT_REG R0, LR0, CR0;;",
            ),
            "execute_fn": "execute_ldr_mult_reg",
        },
        "LDR_CYCLIC_MULT_REG": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
                {"name": "index", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Cyclic Register",
                summary="Load with cyclic addressing into R_CYCLIC.",
                syntax="LDR_CYCLIC_MULT_REG offset, base, index",
                operands=[
                    "offset: Offset register (LR0–LR15)",
                    "base: Base address register (CR0–CR15)",
                    "index: Index inside cyclic register (LR0–LR15)",
                ],
                operation="R_CYCLIC[index % 512:128] = Memory[offset + base]",
            ),
            "execute_fn": "execute_ldr_cyclic_mult_reg",
        },
        "LDR_MULT_MASK_REG": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Load Mask Register",
                summary="Load mask data from memory.",
                syntax="LDR_MULT_MASK_REG offset, base, mask_idx",
                operands=[
                    "offset: Offset register (LR0–LR15)",
                    "base: Base address register (CR0–CR15)",
                ],
                operation="R_MASK = Memory[offset + base]",
            ),
            "execute_fn": "execute_ldr_mult_mask_reg",
        },
        "XMEM_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (XMEM)",
                summary="No operation for xmem slot.",
                syntax="XMEM_NOP",
                operands=[],
            ),
            "execute_fn": "execute_xmem_nop",
        },
        "STR_POST_AAQ_REG": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Store Post-AAQ register",
                summary=(
                    "Write **512 bytes** of **`POST_AAQ_REG`** to external memory. **Interim:** "
                    "the buffer is **512 bytes** (128×32-bit lanes) until quantization export is finalized."
                ),
                syntax="STR_POST_AAQ_REG offset, base",
                operands=[
                    "offset: Offset register (LR0–LR15)",
                    "base: Base address register (CR0–CR15)",
                ],
                operation="Memory[offset + base] = POST_AAQ_REG (512 bytes); interim staging register",
                example="STR_POST_AAQ_REG LR0, CR0;;",
            ),
            "execute_fn": "execute_str_post_aaq_reg",
        },
    },
    
    # =========================================================================
    # LR Slot (Loop Register Instructions)
    # Opcode = position: SET=0, ADD=1, SUB=2, INCR_MOD_POW2=3
    # =========================================================================
    "lr": {
        "SET": {
            "operands": [
                {"name": "reg", "type": "LrIdx"},
                {"name": "src", "type": "CrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Set Loop Register",
                summary="Copy a 32-bit value from a configuration register into a loop register.",
                syntax="SET reg, src",
                operands=[
                    "reg: Loop register (LR0–LR15)",
                    "src: Source configuration register (CR0–CR15)",
                ],
                operation="reg = cr[src]",
                example="SET LR0, CR1;;",
            ),
            "execute_fn": "execute_lr_set",
        },
        "ADD": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "src_a", "type": "LrIdx", "read": "snapshot"},
                {"name": "src_b", "type": "AddSubSrcB", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Add",
                summary=(
                    "Add two sources (second source may be an LR, CR, or 5-bit unsigned immediate) "
                    "and store the result in the destination LR."
                ),
                syntax="ADD dest, src_a, src_b",
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15, CR0–CR15, or unsigned immediate 0–31",
                ],
                operation="dest = src_a + src_b",
                example="ADD LR0, LR1, LR2;;\nADD LR3, LR1, CR5;;\nADD LR4, LR1, 7;;",
            ),
            "execute_fn": "execute_lr_add",
        },
        "SUB": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "src_a", "type": "LrIdx", "read": "snapshot"},
                {"name": "src_b", "type": "AddSubSrcB", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Subtract",
                summary=(
                    "Subtract the second source from the first (second source may be an LR, CR, "
                    "or 5-bit unsigned immediate) and store the result in the destination LR."
                ),
                syntax="SUB dest, src_a, src_b",
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15, CR0–CR15, or unsigned immediate 0–31",
                ],
                operation="dest = src_a - src_b",
                example="SUB LR0, LR1, LR2;;\nSUB LR3, LR1, CR5;;\nSUB LR4, LR1, 7;;",
            ),
            "execute_fn": "execute_lr_sub",
        },
        "INCR_MOD_POW2": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "step", "type": "LcrIdx", "read": "snapshot"},
                {"name": "k", "type": "LrModPow2KImmediate"},
            ],
            "doc": InstructionDoc(
                title="Increment Loop Register Modulo Power of Two",
                summary=(
                    "Add a loop or configuration register into the destination loop "
                    "register, then mask to k low bits (mod 2^k)."
                ),
                syntax="INCR_MOD_POW2 dst, step, k",
                operands=[
                    "dst: Destination loop register (LR0–LR15); read and written",
                    "step: Signed 32-bit increment from LR0–LR15 or CR0–CR15",
                    "k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1",
                ],
                operation="dst <- (dst + step) & ((1 << k) - 1)",
                example="INCR_MOD_POW2 LR2, LR3, 4;;",
            ),
            "execute_fn": "execute_lr_incr_mod_pow2",
        },
    },
    
    # =========================================================================
    # MULT Slot (Multiply Instructions)
    # Opcode = position: MULT.EE=0, MULT.VE.CYCLIC=1, MULT.VE.PADDED=2, MULT_NOP=3,
    #          MULT.VE.CR=4, MULT.VE.AAQ=5, MULT.EE.RR=6
    # =========================================================================
    "mult": {
        "MULT.EE": {
            "operands": [
                {"name": "ra", "type": "MultStageReg", "read": "live"},
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Element-wise Multiply",
                summary="Multiply elements of two registers element by element.",
                syntax="MULT.EE ra, cyclic_offset, mask_offset, mask_shift",
                operands=[
                    "`ra`: **`R0`** | **`R1`** — multiplicand mult-stage register (same cycle as `LDR_MULT_REG` into **`R0`**/**`R1`** is allowed).",
                    "`cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`**.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — shift applied to the mask register.",
                ],
                operation="For each lane i: MULT_RES[i] = ipu_mult(ra[i], R_CYCLIC[cyclic_offset + i]); then apply mask and shift.",
                example="MULT.EE R0, LR0, 0, LR2;;",
            ),
            "execute_fn": "execute_mult_ee",
        },
        "MULT.VE.CYCLIC": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "fixed_idx", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Vector-Element Multiply (cyclic RC)",
                summary=(
                    "Multiply a fixed element from R0 or R1 against R_CYCLIC[cyclic_offset:cyclic_offset+128]. "
                    "`fixed_idx` 0..127 selects `R0[fixed_idx]`, 128..255 selects `R1[fixed_idx - 128]`. "
                    "R_CYCLIC is addressed cyclically modulo 512 elements (no padding with 1 past the boundary)."
                ),
                syntax="MULT.VE.CYCLIC cyclic_offset, mask_offset, mask_shift, fixed_idx",
                operands=[
                    "`cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`** (reduced mod 512).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — shift applied to the mask register.",
                    "`fixed_idx`: **`LR0`**…**`LR15`** (value read live) — scalar index into **`R0`**/**`R1`**.",
                ],
                operation="For i in [0, 128): rb = R_CYCLIC[(cyclic_offset + i) % 512]; scalar from R0/R1 via fixed_idx; MULT_RES[i] = scalar * rb (then mask/shift).",
                example="MULT.VE.CYCLIC LR0, 0, LR2, LR3;;",
            ),
            "execute_fn": "execute_mult_ve_cyclic",
        },
        "MULT.VE.PADDED": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "fixed_idx", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Vector-Element Multiply (padded RC)",
                summary=(
                    "Same scalar × RC row as `MULT.VE.CYCLIC`, but indices at or past the 512-byte RC "
                    "boundary within the 128-element window use a dtype-specific 1 instead of wrapping."
                ),
                syntax="MULT.VE.PADDED cyclic_offset, mask_offset, mask_shift, fixed_idx",
                operands=[
                    "`cyclic_offset`: **`LR0`**…**`LR15`** — base byte offset into `R_CYCLIC`; out-of-range lanes use dtype 1.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in `R_MASK`.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — shift applied to the mask register.",
                    "`fixed_idx`: **`LR0`**…**`LR15`** (value read live) — scalar index into **`R0`**/**`R1`**.",
                ],
                operation="For i in [0, 128): rb = R_CYCLIC[cyclic_offset + i] if in bounds else dtype_one; scalar from R0/R1; MULT_RES[i] = scalar * rb (then mask/shift).",
                example="MULT.VE.PADDED LR0, 0, LR2, LR3;;",
            ),
            "execute_fn": "execute_mult_ve_padded",
        },
        "MULT_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (MULT)",
                summary="No operation for multiply slot.",
                syntax="MULT_NOP",
                operands=[],
            ),
            "execute_fn": "execute_mult_nop",
        },
        "MULT.VE.CR": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "cr_idx", "type": "CrIdx"},
            ],
            "doc": InstructionDoc(
                title="Vector-Element Multiply (CR scalar)",
                summary="Multiply each element of RC[cyclic_offset:cyclic_offset+128] by a scalar from a CR register. Elements beyond RC boundary are treated as 1 (dtype-specific).",
                syntax="MULT.VE.CR cyclic_offset, mask_offset, mask_shift, cr_idx",
                operands=[
                    "cyclic_offset: Base offset into RC (cyclic register); non-cyclic — out-of-bounds elements are padded with 1",
                    "mask_offset: Immediate mask slot 0–7 (128-bit slice of R_MASK)",
                    "mask_shift: Shift applied to the mask (from LR)",
                    "cr_idx: CR register whose low byte supplies the fixed scalar multiplier (CR0–CR15)",
                ],
                operation="For i in [0,128): rb = RC[cyclic_offset+i] if in bounds else dtype_one; MULT_RES[i] = CR[cr_idx][0] * rb",
                example="MULT.VE.CR LR0, 0, LR15, CR3;;",
            ),
            "execute_fn": "execute_mult_ve_cr",
        },
        "MULT.VE.AAQ": {
            "operands": [
                {"name": "cyclic_offset", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Vector-Element Multiply (AAQ scalar)",
                summary="Multiply each element of RC[cyclic_offset:cyclic_offset+128] by a scalar from an AAQ register. Elements beyond RC boundary are treated as 1 (dtype-specific).",
                syntax="MULT.VE.AAQ cyclic_offset, mask_offset, mask_shift, aaq_rf_idx",
                operands=[
                    "cyclic_offset: Base offset into RC (cyclic register); non-cyclic — out-of-bounds elements are padded with 1",
                    "mask_offset: Immediate mask slot 0–7 (128-bit slice of R_MASK)",
                    "mask_shift: Shift applied to the mask (from LR)",
                    "aaq_rf_idx: AAQ register whose low byte supplies the fixed scalar multiplier (AAQ0–AAQ3)",
                ],
                operation="For i in [0,128): rb = RC[cyclic_offset+i] if in bounds else dtype_one; MULT_RES[i] = AAQ[aaq_rf_idx][0] * rb",
                example="MULT.VE.AAQ LR0, 0, LR15, AAQ1;;",
            ),
            "execute_fn": "execute_mult_ve_aaq",
        },
        "MULT.EE.RR": {
            "operands": [
                {"name": "ra", "type": "MultStageReg", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Multi-Element Multiply (register by register)",
                summary=(
                    "Multi-element execution (MEE): multiply a mult-stage register "
                    "element by element against itself. `ra` selects the execution "
                    "mode — **`R0`** gives r0-by-r0, **`R1`** gives r1-by-r1."
                ),
                syntax="MULT.EE.RR ra, mask_offset, mask_shift",
                operands=[
                    "`ra`: **`R0`** | **`R1`** — selects the MEE mode; the chosen register is both multiplicand and multiplier (same cycle as `LDR_MULT_REG` into **`R0`**/**`R1`** is allowed).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`r_mask`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — shift applied to the mask register.",
                ],
                operation="For each lane i: mult_res[i] = ipu_mult(ra[i], ra[i]); then apply mask and shift.",
                example="MULT.EE.RR R0, 0, LR2;;",
            ),
            "execute_fn": "execute_mult_ee_rr",
        },
    },

    # =========================================================================
    # ACC Slot (Accumulator Instructions)
    # Opcode = position: ACC=0, ACC.FIRST=1, RESET_ACC=2, ACC_NOP=3, ACC.ADD_AAQ=4, ACC.ADD_AAQ.FIRST=5, ACC.MAX=6, ACC.MAX.FIRST=7, ACC.STRIDE=8
    # =========================================================================
    "acc": {
        "ACC": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate",
                summary="Accumulate multiply result.",
                syntax="ACC",
                operands=[],
                operation="R_ACC += multiply_result",
            ),
            "execute_fn": "execute_acc",
        },
        "ACC.FIRST": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate First",
                summary="Set accumulator to multiply result (do not ADD to previous R_ACC).",
                syntax="ACC.FIRST",
                operands=[],
                operation="R_ACC = multiply_result",
                example="ACC.FIRST;;",
            ),
            "execute_fn": "execute_acc_first",
        },
        "RESET_ACC": {
            "operands": [],
            "doc": InstructionDoc(
                title="Reset Accumulator",
                summary="Reset accumulator to zero.",
                syntax="RESET_ACC",
                operands=[],
                operation="R_ACC = 0",
            ),
            "execute_fn": "execute_reset_acc",
        },
        "ACC_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (ACC)",
                summary="No operation for accumulator slot.",
                syntax="ACC_NOP",
                operands=[],
            ),
            "execute_fn": "execute_acc_nop",
        },
        "ACC.ADD_AAQ": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulate and Add AAQ",
                summary="Accumulate multiply result, then ADD the selected AAQ register (32-bit) to each of the 128 accumulator words.",
                syntax="ACC.ADD_AAQ aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (AAQ0–AAQ3)",
                ],
                operation=(
                    "R_ACC += multiply_result;\n"
                    "for i in [0, 128): R_ACC[i] += AAQ_REGS[aaq_rf_idx]"
                ),
                example="ACC.ADD_AAQ AAQ0;;",
            ),
            "execute_fn": "execute_acc_add_aaq",
        },
        "ACC.ADD_AAQ.FIRST": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulate and Add AAQ (First)",
                summary="Set accumulator to multiply result plus selected AAQ register (do not ADD to previous R_ACC).",
                syntax="ACC.ADD_AAQ.FIRST aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (AAQ0–AAQ3)",
                ],
                operation=(
                    "R_ACC = multiply_result;\n"
                    "for i in [0, 128): R_ACC[i] += AAQ_REGS[aaq_rf_idx]"
                ),
                example="ACC.ADD_AAQ.FIRST AAQ0;;",
            ),
            "execute_fn": "execute_acc_add_aaq_first",
        },
        "ACC.MAX": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Max",
                summary="For each element, SET R_ACC[i] = max(R_ACC[i], MULT_RES[i], AAQ_REGS[aaq_rf_idx]).",
                syntax="ACC.MAX aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (AAQ0–AAQ3)",
                ],
                operation=(
                    "for i in [0, 128): R_ACC[i] = max(R_ACC[i], MULT_RES[i], AAQ_REGS[aaq_rf_idx])"
                ),
                example="ACC.MAX AAQ0;;",
            ),
            "execute_fn": "execute_acc_max",
        },
        "ACC.MAX.FIRST": {
            "operands": [
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Max (First)",
                summary="For each element, SET R_ACC[i] = max(MULT_RES[i], AAQ_REGS[aaq_rf_idx]). Previous R_ACC is ignored (treated as 0).",
                syntax="ACC.MAX.FIRST aaq_rf_idx",
                operands=[
                    "aaq_rf_idx: AAQ register index (AAQ0–AAQ3)",
                ],
                operation=(
                    "for i in [0, 128): R_ACC[i] = max(MULT_RES[i], AAQ_REGS[aaq_rf_idx])"
                ),
                example="ACC.MAX.FIRST AAQ0;;",
            ),
            "execute_fn": "execute_acc_max_first",
        },
        "ACC.STRIDE": {
            "operands": [
                {"name": "elements_in_row", "type": "ElementsInRow"},
                {"name": "horizontal_stride", "type": "HorizontalStride"},
                {"name": "vertical_stride", "type": "VerticalStride"},
                {"name": "offset", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Stride",
                summary="Reorder the multiplication result into R_ACC using horizontal/vertical stride decimation. Only updates the RACC indexes written; leaves the rest unchanged.",
                syntax="ACC.STRIDE elements_in_row, horizontal_stride, vertical_stride, offset",
                operands=[
                    "elements_in_row: Elements per row (8, 16, 32, or 64)",
                    "horizontal_stride: Horizontal stride mode (enabled, inverted, expand)",
                    "vertical_stride: Vertical stride mode (enabled, inverted)",
                    "offset: LR register; value % 4 gives start index in RACC (0, 32, 64, or 96)",
                ],
                operation=(
                    "Decimate MULT_RES as rows×cols; apply horizontal stride (take every 2nd column, optional expand); "
                    "then vertical stride (take every 2nd row). Write result into R_ACC[start:start+N] where start = (offset%4)*32, N = 32|64|128."
                ),
                example="ACC.STRIDE 8, off, off, LR0;;",
            ),
            "execute_fn": "execute_acc_stride",
        },
    },

    # =========================================================================
    # AAQ Slot (Activation and Quantization)
    # Opcode = position: AAQ_NOP=0, AGG=1, AGG.FIRST=2, AAQ=3, ACTIVATE=4
    # =========================================================================
    "aaq": {
        "AAQ_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (AAQ)",
                summary="No operation for AAQ slot.",
                syntax="AAQ_NOP",
                operands=[],
            ),
            "execute_fn": "execute_aaq_nop",
        },
        "AGG": {
            "operands": [
                {"name": "agg_mode", "type": "AggMode"},
                {"name": "post_fn", "type": "PostFn"},
                {
                    "name": "valid_elements",
                    "type": "LcrIdx",
                    "read": "snapshot",
                },
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Aggregate",
                summary=(
                    "Collapse R_ACC lanes into one value (SUM or MAX), using only the first "
                    "valid_elements words for the tree; apply post function; store to selected AAQ register."
                ),
                syntax="AGG agg_mode, post_fn, valid_elements, cr_idx, aaq_rf_idx",
                operands=[
                    "agg_mode: sum or max",
                    "post_fn: value, value_cr, inv, or inv_sqrt",
                    "valid_elements: lane count from an LR or CR register (value read at cycle start; "
                    "unsigned, clamped to 0–128; indices [valid_elements..127] are masked out)",
                    "cr_idx: CR register for value_cr post function (CR0–CR15)",
                    "aaq_rf_idx: AAQ register to store result (AAQ0–AAQ3)",
                ],
                operation=(
                    "Let n = min(valid_elements, 128). "
                    "If sum: v = sum(R_ACC[0..n-1]). "
                    "If max: v = max(R_ACC[0..n-1], AAQ[aaq_rf_idx]). "
                    "Apply post_fn(v): value→v, value_cr→v*cr[cr_idx], inv→1/v, inv_sqrt→1/sqrt(v). "
                    "AAQ[aaq_rf_idx] = result."
                ),
                example="AGG sum, value, LR0, CR0, AAQ0;;",
            ),
            "execute_fn": "execute_agg",
        },
        "AGG.FIRST": {
            "operands": [
                {"name": "agg_mode", "type": "AggMode"},
                {"name": "post_fn", "type": "PostFn"},
                {
                    "name": "valid_elements",
                    "type": "LcrIdx",
                    "read": "snapshot",
                },
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "aaq_rf_idx", "type": "AaqRegIdx"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Aggregate First",
                summary="Like AGG, but for MAX mode ignores the previous AAQ register value, avoiding contamination from uninitialized data.",
                syntax="AGG.FIRST agg_mode, post_fn, valid_elements, cr_idx, aaq_rf_idx",
                operands=[
                    "agg_mode: sum or max",
                    "post_fn: value, value_cr, inv, or inv_sqrt",
                    "valid_elements: lane count from an LR or CR register (value read at cycle start; "
                    "unsigned, clamped to 0–128; indices [valid_elements..127] are masked out)",
                    "cr_idx: CR register for value_cr post function (CR0–CR15)",
                    "aaq_rf_idx: AAQ register to store result (AAQ0–AAQ3)",
                ],
                operation=(
                    "Let n = min(valid_elements, 128). "
                    "If sum: v = sum(R_ACC[0..n-1]). "
                    "If max: v = max(R_ACC[0..n-1]) (previous AAQ value is NOT included). "
                    "Apply post_fn(v): value→v, value_cr→v*cr[cr_idx], inv→1/v, inv_sqrt→1/sqrt(v). "
                    "AAQ[aaq_rf_idx] = result."
                ),
                example="AGG.FIRST max, value, LR0, CR0, AAQ0;;",
            ),
            "execute_fn": "execute_agg_first",
        },
        "AAQ": {
            "operands": [],
            "doc": InstructionDoc(
                title="AAQ Quantize",
                summary=(
                    "Quantize the 128 wide lanes in **`POST_AAQ_REG`** (INT32 per lane in INT8 mode) "
                    "to INT8, storing clamped bytes in the **leading 128 bytes** of **`POST_AAQ_REG`** "
                    "and clearing the rest of the register. Wide lanes are normally produced by "
                    "**`ACTIVATE`** (from ``r_acc``). Requires INT8 mode."
                ),
                syntax="AAQ",
                operands=[],
                operation=(
                    "Requires INT8 mode (CR15 == DType.INT8). "
                    "For i in [0, 128): POST_AAQ_REG[i] = clamp(trunc(POST_AAQ_REG wide lane i), -128, 127); "
                    "POST_AAQ_REG[128..511] = 0"
                ),
                example="AAQ;;",
            ),
            "execute_fn": "execute_aaq",
        },
        "ACTIVATE": {
            "operands": [
                {
                    "name": "valid_elements",
                    "type": "LcrIdx",
                    "read": "snapshot",
                },
                {"name": "activation_fn", "type": "ActivationFn"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Activation",
                summary=(
                    "Read each of the first ``valid_elements`` lanes from ``r_acc``, apply the "
                    "selected element-wise activation, and write results into the same lane indices "
                    "of ``POST_AAQ_REG`` (``r_acc`` is unchanged). The activation is "
                    "selected by keyword (see ACTIVATION_FN_NAMES). Behaviour matches the activation "
                    "table in the AAQ stage spec (section 7.0). The selector uses four bits; "
                    "encodings outside the twelve named activations behave as identity. For Python "
                    "emulator calibration (virtual α), see "
                    "docs/content/building-applications.md#activations-emulator."
                ),
                syntax="ACTIVATE valid_elements activation_fn",
                operands=[
                    "valid_elements: lane count from an LR or CR register (unsigned, clamped to 0–128)",
                    "activation_fn: keyword naming the activation (e.g. relu, gelu, silu; see ACTIVATION_FN_NAMES)",
                ],
                operation=(
                    "Let n = min(valid_elements, 128) and k = encoded activation index. "
                    "For i in [0, n): POST_AAQ_REG[i] = activation_k(R_ACC[i]) (same 32-bit lane format as R_ACC). "
                    "R_ACC is not modified. The selector uses four bits; encodings outside the twelve named "
                    "activations behave as identity. "
                    "α for leaky_relu, elu, and prelu is not an ISA operand; see "
                    "docs/content/building-applications.md#activations-emulator."
                ),
                example="ACTIVATE LR0 relu;;",
            ),
            "execute_fn": "execute_activate",
        },
    },

    # =========================================================================
    # COND Slot (Conditional Branch Instructions)
    # Opcode = position: BEQ=0, BNE=1, BLT=2, BNZ=3, BZ=4, B=5, BR=6, BKPT=7
    # =========================================================================
    "cond": {
        "BEQ": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Equal",
                summary="Branch if two registers are equal.",
                syntax="BEQ reg1, reg2, label",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 == reg2) PC = label",
                example="BEQ LR0, LR1, end;;",
            ),
            "execute_fn": "execute_beq",
        },
        "BNE": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Not Equal",
                summary="Branch if two registers are not equal.",
                syntax="BNE reg1, reg2, label",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 != reg2) PC = label",
                example="BNE LR0, CR0, loop;;",
            ),
            "execute_fn": "execute_bne",
        },
        "BLT": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Less Than",
                summary="Branch if first register is less than second.",
                syntax="BLT reg1, reg2, label",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 < reg2) PC = label",
                example="BLT LR0, CR1, smaller;;",
            ),
            "execute_fn": "execute_blt",
        },
        "BNZ": {
            "operands": [
                {"name": "test_reg", "type": "LcrIdx", "read": "snapshot"},
                {"name": "base_reg", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Not Zero",
                summary="Branch if test register not equal to base register.",
                syntax="BNZ test_reg, base_reg, label",
                operands=[
                    "test_reg: Register to test (LR0–LR15 or CR0–CR15)",
                    "base_reg: Base comparison register (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (test_reg != base_reg) PC = label",
                example="BNZ LR3, LR0, loop;;",
            ),
            "execute_fn": "execute_bnz",
        },
        "BZ": {
            "operands": [
                {"name": "test_reg", "type": "LcrIdx", "read": "snapshot"},
                {"name": "base_reg", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Zero",
                summary="Branch if test register equals base register.",
                syntax="BZ test_reg, base_reg, label",
                operands=[
                    "test_reg: Register to test (LR0–LR15 or CR0–CR15)",
                    "base_reg: Base comparison register (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (test_reg == base_reg) PC = label",
                example="BZ LR0, LR1, zero;;",
            ),
            "execute_fn": "execute_bz",
        },
        "B": {
            "operands": [
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Unconditional Branch",
                summary="Always branch to label.",
                syntax="B label",
                operands=["label: Branch target label"],
                operation="PC = label",
                example="B start;;",
            ),
            "execute_fn": "execute_b",
        },
        "BR": {
            "operands": [
                {"name": "reg", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Branch Register",
                summary="Branch to address in register.",
                syntax="BR reg",
                operands=["reg: Register containing target address (LR0–LR15 or CR0–CR15)"],
                operation="PC = reg",
            ),
            "execute_fn": "execute_br",
        },
        "BKPT": {
            "operands": [],
            "doc": InstructionDoc(
                title="Breakpoint",
                summary="Conditional breakpoint.",
                syntax="BKPT",
                operands=[],
                operation="Halt execution (debugging)",
            ),
            "execute_fn": "execute_bkpt",
        },
    },

    # =========================================================================
    # BREAK Slot (Break Instructions)
    # Opcode = position: BREAK=0, BREAK.IFEQ=1, BREAK_NOP=2
    # =========================================================================
    "break": {
        "BREAK": {
            "operands": [],
            "doc": InstructionDoc(
                title="Break",
                summary="Unconditional break.",
                syntax="BREAK",
                operands=[],
                operation="Halt execution",
            ),
            "execute_fn": "execute_break",
        },
        "BREAK.IFEQ": {
            "operands": [
                {"name": "reg", "type": "LrIdx", "read": "snapshot"},
                {"name": "value", "type": "BreakImmediate"},
            ],
            "doc": InstructionDoc(
                title="Break if Equal",
                summary="Break execution if register equals value.",
                syntax="BREAK.IFEQ reg, value",
                operands=[
                    "reg: Register to test (LR0–LR15)",
                    "value: Immediate value to compare against",
                ],
                operation="if (reg == value) BREAK",
                example="BREAK.IFEQ LR0, 10;;",
            ),
            "execute_fn": "execute_break_ifeq",
        },
        "BREAK_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (BREAK)",
                summary="No operation for BREAK slot.",
                syntax="BREAK_NOP",
                operands=[],
            ),
            "execute_fn": "execute_break_nop",
        },
    },
}


# ===========================================================================
# Query Functions (position-based opcode derivation)
# ===========================================================================

def canonical_instruction_name(slot_type: str, instruction_name: str) -> str:
    """Return the canonical instruction key for ``instruction_name`` (case-insensitive)."""
    instructions = INSTRUCTION_SPEC[slot_type]
    lowered = instruction_name.lower()
    for name in instructions:
        if name.lower() == lowered:
            return name
    raise KeyError(
        f"Instruction {instruction_name!r} not found in slot {slot_type!r}"
    )


def get_opcode_for_instruction(slot_type: str, instruction_name: str) -> int:
    """Get opcode index for an instruction (derived from its position).
    
    Args:
        slot_type: Slot type (e.g., "xmem", "lr", "mult")
        instruction_name: Instruction name (e.g., "STR_ACC_REG")
    
    Returns:
        Opcode index (0-based position in slot's instruction list)
    
    Raises:
        KeyError if slot or instruction not found
    """
    canon = canonical_instruction_name(slot_type, instruction_name)
    instructions = INSTRUCTION_SPEC[slot_type]
    for idx, name in enumerate(instructions.keys()):
        if name == canon:
            return idx
    raise KeyError(f"Instruction '{instruction_name}' not found in slot '{slot_type}'")


def extract_opcodes() -> Dict[str, List[str]]:
    """Extract all instruction names grouped by slot type (in order).
    
    Returns a dict mapping slot type to list of instruction names, where
    the position in the list IS the opcode index.
    
    Example:
        {
            "xmem": ["STR_ACC_REG", "LDR_MULT_REG", ...],
            "lr": ["SET", "ADD", "SUB", "INCR_MOD_POW2"],
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
        instruction_name: Instruction name (e.g., "STR_ACC_REG")
    
    Returns:
        The instruction definition dict with "operands", "doc", "execute_fn"
    
    Raises:
        KeyError if instruction not found
    """
    return INSTRUCTION_SPEC[slot_type][canonical_instruction_name(slot_type, instruction_name)]


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
            tv = opcode_token.value.lower()
            for subclass in cls.__subclasses__():
                if any(tv == name.lower() for name in subclass.enum_array()):
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
            # Convert name to constant style: "STR_ACC_REG" → "STR_ACC_REG"
            const_name = instruction_name.upper().replace(".", "_").replace("-", "_")
            constants[f"{prefix}_{const_name}"] = opcode_idx
        
        # Add count constant
        constants[f"NUM_{prefix}"] = len(instructions)
    
    return constants


# ===========================================================================
# Operand types (single source for validation + documentation generation)
# ===========================================================================

VALID_OPERAND_TYPES: frozenset[str] = frozenset(
    {
        "MultStageReg",
        "LrIdx",
        "CrIdx",
        "LcrIdx",
        "AaqRegIdx",
        "ElementsInRow",
        "HorizontalStride",
        "VerticalStride",
        "AggMode",
        "PostFn",
        "LrModPow2KImmediate",
        "MultMaskOffsetImmediate",
        "ActivationFn",
        "BreakImmediate",
        "Label",
        "AddSubSrcB",
    }
)


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
    valid_operand_types = VALID_OPERAND_TYPES
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

# Derive union layout from INSTRUCTION_SPEC (must run after spec is fully defined).
from ipu_common.union_layout import compute_slot_layouts, SlotUnion  # noqa: E402

_computed_unions = compute_slot_layouts(INSTRUCTION_SPEC)
SLOT_UNIONS.update(_computed_unions)
SLOT_BINARY_LAYOUT.update({
    slot: [f.canonical_type for f in su.fields]
    for slot, su in _computed_unions.items()
})
