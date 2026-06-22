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
  - "CrIdx": CR0–CR14 (CrRegField)
  - "LcrIdx": LR0–LR15 or CR0–CR14 (LcrRegField)
  - "LrIncDecImmediate": unsigned immediate for INC/DEC; bit width derived from LR slot union layout
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
    "SLOT_METADATA",
    "COMPOUND_LAYOUT_SLOT_ORDER",
    "VALID_OPERAND_TYPES",
    "SlotUnion",
    "canonical_instruction_name",
    "get_instruction",
    "get_instruction_by_opcode",
    "create_assembler_opcodes",
    "create_emulator_constants",
    "get_operand_names_and_types",
    "is_hardware_slot",
    "validate_instruction_spec",
    "PSEUDO_INSTRUCTION_SPEC",
    "find_pseudo_instruction",
    "validate_pseudo_instruction_spec",
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
    notes: str | None = None


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
    "load": 1,
    "store": 1,
    "acc_store": 1,
    "mult": 1,
    "acc": 1,
    "aaq": 1,
    "lr": 3,
    "cond": 1,
}

# Compound instruction binary layout (MSB → LSB).  Used by the assembler,
# union-layout SVG, and docs — keep in sync with compound_inst encoding order.
COMPOUND_LAYOUT_SLOT_ORDER: list[str] = [
    "cond",
    "lr",
    "load",
    "mult",
    "acc",
    "aaq",
    "store",
    "acc_store",
    "break",
]

# Per-slot metadata.  ``hardware: False`` marks simulation-only slots that are
# not implemented in real IPU hardware (excluded from HW codegen).
SLOT_METADATA: dict[str, dict[str, bool]] = {
    "acc_store": {"hardware": False},
}


def is_hardware_slot(slot_type: str) -> bool:
    """Return True when *slot_type* is implemented in real IPU hardware."""
    return SLOT_METADATA.get(slot_type, {}).get("hardware", True)


# ===========================================================================
# MASTER INSTRUCTION SPECIFICATION
# ===========================================================================
# Each slot type (load, store, acc_store, lr, mult, acc, cond, break) is defined
# separately.  Instructions maintain ORDER — position in dict determines opcode!
# ===========================================================================

INSTRUCTION_SPEC = {
    # =========================================================================
    # LOAD Slot (first pipeline stage — feeds the multiply unit)
    # Opcode = position in list: LDR_MULT_REG=0, LDR_CYCLIC_MULT_REG=1, etc.
    # =========================================================================
    "load": {
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
                    "base: Base address register (CR0–CR14)",
                    "index: Index inside cyclic register (LR0–LR15); must hold 0, 128, 256, or 384 — the four R_CYCLIC slot boundaries. Any other value raises an error.",
                ],
                operation="R_CYCLIC[index .. index+127] = Memory[offset + base]   # index ∈ {0, 128, 256, 384}",
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
                    "base: Base address register (CR0–CR14)",
                ],
                operation="R_MASK = Memory[offset + base]",
            ),
            "execute_fn": "execute_ldr_mult_mask_reg",
        },
        "LOAD_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (LOAD)",
                summary="No operation for load slot.",
                syntax="LOAD_NOP",
                operands=[],
            ),
            "execute_fn": "execute_load_nop",
        },
    },

    # =========================================================================
    # STORE Slot (last pipeline stage — drains POST_AAQ_REG after AAQ/activate)
    # =========================================================================
    "store": {
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
                    "base: Base address register (CR0–CR14)",
                ],
                operation="Memory[offset + base] = POST_AAQ_REG (512 bytes); interim staging register",
                example="STR_POST_AAQ_REG LR0, CR0;;",
            ),
            "execute_fn": "execute_str_post_aaq_reg",
        },
        "STORE_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (STORE)",
                summary="No operation for store slot.",
                syntax="STORE_NOP",
                operands=[],
            ),
            "execute_fn": "execute_store_nop",
        },
    },

    # =========================================================================
    # ACC_STORE Slot (simulation-only — NOT implemented in real IPU hardware)
    # =========================================================================
    "acc_store": {
        "STR_ACC_REG": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "live"},
                {"name": "base", "type": "CrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Store Accumulator",
                summary=(
                    "Store **R_ACC** to external memory. **Simulation-only** — not "
                    "implemented in real IPU hardware."
                ),
                syntax="STR_ACC_REG offset, base",
                operands=[
                    "offset: Offset register (LR0–LR15)",
                    "base: Base address register (CR0–CR14)",
                ],
                operation="Memory[offset + base] = R_ACC",
                example="STR_ACC_REG CR0, CR1;;",
                notes="This instruction lives in the simulation-only **acc_store** slot.",
            ),
            "execute_fn": "execute_str_acc_reg",
        },
        "ACC_STORE_NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation (ACC_STORE)",
                summary="No operation for acc_store slot (simulation-only).",
                syntax="ACC_STORE_NOP",
                operands=[],
            ),
            "execute_fn": "execute_acc_store_nop",
        },
    },
    
    # =========================================================================
    # LR Slot (Loop Register Instructions)
    # Opcode = position: SET=0, ADD=1, SUB=2, INCR_MOD_POW2=3, INC=4, DEC=5
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
                    "src: Source configuration register (CR0–CR14)",
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
                {"name": "src_b", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Add",
                summary=(
                    "Add two register sources and store the result in the destination LR."
                ),
                syntax="ADD dest, src_a, src_b",
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15 or CR0–CR14",
                ],
                operation="dest = src_a + src_b",
                example="ADD LR0, LR1, LR2;;\nADD LR3, LR1, CR5;;",
            ),
            "execute_fn": "execute_lr_add",
        },
        "SUB": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "src_a", "type": "LrIdx", "read": "snapshot"},
                {"name": "src_b", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Subtract",
                summary=(
                    "Subtract the second source from the first and store the result "
                    "in the destination LR."
                ),
                syntax="SUB dest, src_a, src_b",
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15 or CR0–CR14",
                ],
                operation="dest = src_a - src_b",
                example="SUB LR0, LR1, LR2;;\nSUB LR3, LR1, CR5;;",
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
                    "step: Signed 32-bit increment from LR0–LR15 or CR0–CR14",
                    "k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1",
                ],
                operation="dst <- (dst + step) & ((1 << k) - 1)",
                example="INCR_MOD_POW2 LR2, LR3, 4;;",
            ),
            "execute_fn": "execute_lr_incr_mod_pow2",
        },
        "INC": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "imm", "type": "LrIncDecImmediate"},
            ],
            "doc": InstructionDoc(
                title="Increment",
                summary=(
                    "Add an unsigned immediate to the destination LR (read-modify-write)."
                ),
                syntax="INC dest, imm",
                operands=[
                    "dest: Destination local register (LR0–LR15); also the implicit source",
                    "imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout",
                ],
                operation="dest = dest + imm",
                example="INC LR0, 7;;",
            ),
            "execute_fn": "execute_lr_inc",
        },
        "DEC": {
            "operands": [
                {"name": "dest", "type": "LrIdx"},
                {"name": "imm", "type": "LrIncDecImmediate"},
            ],
            "doc": InstructionDoc(
                title="Decrement",
                summary=(
                    "Subtract an unsigned immediate from the destination LR (read-modify-write)."
                ),
                syntax="DEC dest, imm",
                operands=[
                    "dest: Destination local register (LR0–LR15); also the implicit source",
                    "imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout",
                ],
                operation="dest = dest - imm",
                example="DEC LR0, 3;;",
            ),
            "execute_fn": "execute_lr_dec",
        },
    },
    
    # =========================================================================
    # MULT Slot (Multiply Instructions)
    # Opcode = position: MULT.RC.VV=0, MULT.RC.VE=1, MULT.RC.VS=2, MULT_NOP=3,
    #          MULT.VE=4, MULT.EE=5
    # =========================================================================
    "mult": {
        "MULT.RC.VV": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "live"},
                {"name": "ra", "type": "MultStageReg", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="RC Vector × Ra Vector Multiply",
                summary="Multiply R_CYCLIC[rc_idx:rc_idx+128] by Ra (R0 or R1) element-wise.",
                syntax="MULT.RC.VV rc_idx, ra, mask_offset, mask_shift",
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`** (cyclic, mod 512).",
                    "`ra`: **`R0`** | **`R1`** — multiplicand mult-stage register (same cycle as `LDR_MULT_REG` into **`R0`**/**`R1`** is allowed).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                ],
                operation="For each lane i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], ra[i]); then apply mask and shift.",
                example="MULT.RC.VV LR0, R0, 0, LR2;;",
                notes="Lane masking via `mask_offset` and `mask_shift` zeroes lanes whose derived mask bit is **0** (deactivated) in `MULT_RES` before accumulation; lanes with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "execute_fn": "execute_mult_rc_vv",
        },
        "MULT.RC.VE": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "live"},
                {"name": "src", "type": "LcrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="RC Vector × Scalar Multiply",
                summary=(
                    "Multiply R_CYCLIC[rc_idx:rc_idx+128] by a scalar, element-wise. The scalar is "
                    "either a single element of R0/R1 (selected by an LR value) or a CR register value."
                ),
                syntax="MULT.RC.VE rc_idx, src, mask_offset, mask_shift",
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`** (cyclic, mod 512).",
                    "`src`: **`LR0`**…**`LR15`** | **`CR0`**…**`CR14`** — if an LR, its stored value selects the scalar from R0/R1 (0..127 → `R0[idx]`, 128..255 → `R1[idx - 128]`); if a CR, its low byte supplies the scalar directly.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                ],
                operation="For each lane i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], src_value); then apply mask and shift.",
                example="MULT.RC.VE LR0, LR3, 0, LR2;;",
                notes="Lane masking via `mask_offset` and `mask_shift` zeroes lanes whose derived mask bit is **0** (deactivated) in `MULT_RES` before accumulation; lanes with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "execute_fn": "execute_mult_rc_ve",
        },
        "MULT.RC.VS": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "live"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="RC Vector Self-Multiply (Square)",
                summary="Square R_CYCLIC[rc_idx:rc_idx+128] element-wise.",
                syntax="MULT.RC.VS rc_idx, mask_offset, mask_shift",
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base byte offset into **`R_CYCLIC`** (cyclic, mod 512).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                ],
                operation="For each lane i: rb = R_CYCLIC[(rc_idx + i) % 512]; MULT_RES[i] = ipu_mult(rb, rb); then apply mask and shift.",
                example="MULT.RC.VS LR0, 0, LR2;;",
                notes="Lane masking via `mask_offset` and `mask_shift` zeroes lanes whose derived mask bit is **0** (deactivated) in `MULT_RES` before accumulation; lanes with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "execute_fn": "execute_mult_rc_vs",
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
        "MULT.VE": {
            "operands": [
                {"name": "ra_idx", "type": "LrIdx", "read": "live"},
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Ra Vector × CR Scalar Multiply",
                summary="Multiply Ra[ra_idx:ra_idx+128] (combined R0/R1) by a CR scalar, element-wise.",
                syntax="MULT.VE ra_idx, cr_idx, mask_offset, mask_shift",
                operands=[
                    "`ra_idx`: **`LR0`**…**`LR15`** — base byte offset into combined Ra (`R0` ++ `R1`, 256 bytes, cyclic mod 256).",
                    "`cr_idx`: **`CR0`**…**`CR14`** — CR register whose low byte supplies the scalar multiplier.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                ],
                operation="For each lane i: ra = Ra[(ra_idx + i) % 256]; MULT_RES[i] = ipu_mult(ra, CR[cr_idx][0]); then apply mask and shift.",
                example="MULT.VE LR0, CR3, 0, LR2;;",
                notes="Lane masking via `mask_offset` and `mask_shift` zeroes lanes whose derived mask bit is **0** (deactivated) in `MULT_RES` before accumulation; lanes with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "execute_fn": "execute_mult_ve",
        },
        "MULT.EE": {
            "operands": [
                {"name": "ra_idx", "type": "LrIdx", "read": "live"},
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "live"},
            ],
            "doc": InstructionDoc(
                title="Ra Element × CR Scalar Multiply (broadcast)",
                summary="Multiply a single Ra element by a CR scalar; broadcast the product to all 128 lanes.",
                syntax="MULT.EE ra_idx, cr_idx, mask_offset, mask_shift",
                operands=[
                    "`ra_idx`: **`LR0`**…**`LR15`** — index of the single Ra element to read (combined `R0` ++ `R1`, 256 bytes, mod 256).",
                    "`cr_idx`: **`CR0`**…**`CR14`** — CR register whose low byte supplies the scalar multiplier.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                ],
                operation="ra = Ra[ra_idx % 256]; result = ipu_mult(ra, CR[cr_idx][0]); for each lane i: MULT_RES[i] = result; then apply mask and shift.",
                example="MULT.EE LR0, CR3, 0, LR2;;",
                notes="Lane masking via `mask_offset` and `mask_shift` zeroes lanes whose derived mask bit is **0** (deactivated) in `MULT_RES` before accumulation; lanes with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "execute_fn": "execute_mult_ee",
        },
    },

    # =========================================================================
    # ACC Slot (Accumulator Instructions)
    # Opcode = position: ACC=0, ACC.FIRST=1, ACC_NOP=2, ACC.STRIDE=3, AGG.SUM.FIRST=4, AGG.SUM=5, AGG.MAX.FIRST=6, AGG.MAX=7
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
                    "elements_in_row: Elements per row (16, 32, or 64; minimum is 16)",
                    "horizontal_stride: Horizontal stride mode (off, on, on_inv)",
                    "vertical_stride: Vertical stride mode (enabled, inverted)",
                    "offset: LR register; value % 4 gives start index in RACC (0, 32, 64, or 96)",
                ],
                operation=(
                    "Decimate MULT_RES as rows×cols; apply horizontal stride (take every 2nd column); "
                    "then vertical stride (take every 2nd row). Write result into R_ACC[start:start+N] where start = (offset%4)*32, N = 32|64|128. "
                    "When a data structure has fewer than 8 elements, hardware pads to 16 automatically (not programmable)."
                ),
                example="ACC.STRIDE 16, off, off, LR0;;",
            ),
            "execute_fn": "execute_acc_stride",
        },
        "AGG.SUM.FIRST": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "snapshot"},
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Sum (First)",
                summary=(
                    "Sum active R_ACC lanes and write the result into R_ACC at the slot given by LR. "
                    "The current value at the destination slot is NOT included in the sum (clean initialisation). "
                    "``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements."
                ),
                syntax="AGG.SUM.FIRST dest_slot, full_xmem_row",
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements",
                ],
                operation=(
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = sum(R_ACC[0..n-1])."
                ),
                example="AGG.SUM.FIRST LR0, 0;;",
            ),
            "execute_fn": "execute_agg_sum_first",
        },
        "AGG.SUM": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "snapshot"},
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Sum",
                summary=(
                    "Sum active R_ACC lanes and ADD the result to R_ACC at the slot given by LR "
                    "(running cross-cycle accumulation). "
                    "``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements."
                ),
                syntax="AGG.SUM dest_slot, full_xmem_row",
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements",
                ],
                operation=(
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = sum(R_ACC[0..n-1]) + R_ACC[dest]."
                ),
                example="AGG.SUM LR0, 0;;",
            ),
            "execute_fn": "execute_agg_sum",
        },
        "AGG.MAX.FIRST": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "snapshot"},
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Max (First)",
                summary=(
                    "Find the maximum of active R_ACC lanes and write it into R_ACC at the slot given by LR. "
                    "The current value at the destination slot is NOT used as a seed (clean initialisation). "
                    "``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements."
                ),
                syntax="AGG.MAX.FIRST dest_slot, full_xmem_row",
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements",
                ],
                operation=(
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = max(R_ACC[0..n-1]); when n = 0 the identity seed "
                    "(INT32_MIN for integer lanes, -inf for float lanes) is written."
                ),
                example="AGG.MAX.FIRST LR0, 0;;",
            ),
            "execute_fn": "execute_agg_max_first",
        },
        "AGG.MAX": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "snapshot"},
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Max",
                summary=(
                    "Find the maximum of active R_ACC lanes seeded with the current destination slot value "
                    "(running cross-cycle max). "
                    "``full_xmem_row=1`` always uses 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements."
                ),
                syntax="AGG.MAX dest_slot, full_xmem_row",
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements",
                ],
                operation=(
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = max(R_ACC[0..n-1], R_ACC[dest])."
                ),
                example="AGG.MAX LR0, 0;;",
            ),
            "execute_fn": "execute_agg_max",
        },
    },

    # =========================================================================
    # AAQ Slot (Activation and Quantization)
    # Opcode = position: AAQ_NOP=0, AAQ=1, ACTIVATE=2
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
        "AAQ": {
            "operands": [
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="AAQ Quantize",
                summary=(
                    "Quantize wide lanes in **`POST_AAQ_REG`** (INT32 per lane in INT8 mode) "
                    "to INT8, storing clamped bytes in the **leading 128 bytes** of **`POST_AAQ_REG`** "
                    "and clearing the rest of the register. Wide lanes are normally produced by "
                    "**`ACTIVATE`** (from ``r_acc``). Requires INT8 mode. "
                    "``full_xmem_row=1`` always processes all 128 lanes; "
                    "``full_xmem_row=0`` uses ``CR15.valid_elements`` as the active lane count."
                ),
                syntax="AAQ full_xmem_row",
                operands=[
                    "full_xmem_row: 1 = always 128 lanes (full XMEM row); 0 = use CR15.valid_elements lane count",
                ],
                operation=(
                    "Requires INT8 mode (IpuState.dtype == DType.INT8 in the Python emulator). "
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128). "
                    "For i in [0, n): POST_AAQ_REG[i] = clamp(POST_AAQ_REG wide lane i, -128, 127) "
                    "(interim direct INT8 clamp of the post-ACTIVATE lane; a future per-128-element "
                    "requantize will scale lanes into INT8 range before this step and supersede the clamp). "
                    "POST_AAQ_REG[n..511] = 0."
                ),
                example="AAQ 1;;",
            ),
            "execute_fn": "execute_aaq",
        },
        "ACTIVATE": {
            "operands": [
                {"name": "activation_fn", "type": "ActivationFn"},
                {"name": "full_xmem_row", "type": "FullXmemRow"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Activation",
                summary=(
                    "Read active lanes from ``r_acc``, apply the selected element-wise activation, "
                    "and write results into the same lane indices of ``POST_AAQ_REG`` (``r_acc`` is unchanged). "
                    "``full_xmem_row=1`` always activates all 128 lanes; ``full_xmem_row=0`` uses CR15.valid_elements. "
                    "The activation is selected by keyword (see ACTIVATION_FN_NAMES). The available "
                    "activation functions are: ``identity`` (0), ``relu`` (1), ``relu6`` (2), "
                    "``sigmoid`` (3), ``tanh`` (4), ``gelu`` (5), ``softplus`` (6), ``elu`` (7), "
                    "``exp2`` (8), ``reciprocal`` (9), ``rsqrt`` (10). For Python emulator calibration (virtual α), see "
                    "docs/content/building-applications.md#activations-emulator."
                ),
                syntax="ACTIVATE activation_fn, full_xmem_row",
                operands=[
                    "activation_fn: keyword naming the activation (one of identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2; see ACTIVATION_FN_NAMES)",
                    "full_xmem_row: 1 = always 128 lanes; 0 = use CR15.valid_elements (default 0)",
                ],
                operation=(
                    "Let n = 128 if full_xmem_row else min(CR15.valid_elements, 128) "
                    "and k = encoded activation index. "
                    "For i in [0, n): POST_AAQ_REG[i] = activation_k(R_ACC[i]) (same 32-bit lane format as R_ACC). "
                    "R_ACC is not modified. The selector uses four bits; encodings outside the eleven named "
                    "activations behave as identity. "
                    "α for elu is not an ISA operand; see "
                    "docs/content/building-applications.md#activations-emulator."
                ),
                example="ACTIVATE relu, 0;;",
            ),
            "execute_fn": "execute_activate",
        },
    },

    # =========================================================================
    # COND Slot (Conditional Branch Instructions)
    # Opcode = position: BEQ=0, BNE=1, BLT=2, BGE=3, BR=4, BKPT=5
    # BNZ, BZ, and B are pseudo-instructions (see PSEUDO_INSTRUCTION_SPEC):
    # they're each exactly expressible via BEQ/BNE against CR0 (always zero),
    # so they don't need their own opcode.
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
                    "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
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
                    "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
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
                    "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
                    "label: Branch target label",
                ],
                operation="if (reg1 < reg2) PC = label",
                example="BLT LR0, CR1, smaller;;",
            ),
            "execute_fn": "execute_blt",
        },
        "BGE": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "snapshot"},
                {"name": "reg2", "type": "LcrIdx", "read": "snapshot"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Greater or Equal",
                summary="Branch if first register is greater than or equal to second.",
                syntax="BGE reg1, reg2, label",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
                    "label: Branch target label",
                ],
                operation="if (reg1 >= reg2) PC = label",
                example="BGE LR0, CR1, ge;;",
            ),
            "execute_fn": "execute_bge",
        },
        "BR": {
            "operands": [
                {"name": "reg", "type": "LcrIdx", "read": "snapshot"},
            ],
            "doc": InstructionDoc(
                title="Branch Register",
                summary="Branch to address in register.",
                syntax="BR reg",
                operands=["reg: Register containing target address (LR0–LR15 or CR0–CR14)"],
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
        slot_type: Slot type (e.g., "load", "store", "lr", "mult")
        instruction_name: Instruction name (e.g., "LDR_MULT_REG")
    
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
            "load": ["LDR_MULT_REG", "LDR_CYCLIC_MULT_REG", ...],
            "lr": ["SET", "ADD", "SUB", "INCR_MOD_POW2", "INC", "DEC"],
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
        slot_type: Slot type (e.g., "load", "store", "lr", "mult")
        instruction_name: Instruction name (e.g., "LDR_MULT_REG")
    
    Returns:
        The instruction definition dict with "operands", "doc", "execute_fn"
    
    Raises:
        KeyError if instruction not found
    """
    return INSTRUCTION_SPEC[slot_type][canonical_instruction_name(slot_type, instruction_name)]


def get_instruction_by_opcode(slot_type: str, opcode_index: int) -> Tuple[str, dict]:
    """Look up instruction by slot type and opcode (position-based).
    
    Args:
        slot_type: Slot type (e.g., "load", "lr")
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
            "LoadInstOpcode": <class>,
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
        "load": "LoadInstOpcode",
        "store": "StoreInstOpcode",
        "acc_store": "AccStoreInstOpcode",
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
        "load": "LOAD_OP",
        "store": "STORE_OP",
        "acc_store": "ACC_STORE_OP",
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
        "ElementsInRow",
        "HorizontalStride",
        "VerticalStride",
        "LrModPow2KImmediate",
        "LrIncDecImmediate",
        "MultMaskOffsetImmediate",
        "ActivationFn",
        "BreakImmediate",
        "Label",
        "FullXmemRow",
    }
)


# ===========================================================================
# Pseudo-Instruction Specification (assembler-only — no opcode, no execute_fn)
# ===========================================================================
#
# Pseudo-instructions are aliases that the assembler expands into a real
# INSTRUCTION_SPEC entry at compile time. They are NEVER assigned an opcode
# and NEVER appear in the binary, so they need no emulator handler — only
# an "expands_to" mapping onto an existing real instruction.
#
# Looked up by (name, operand count): this lets a pseudo-instruction share
# a name with a real instruction of a different arity without ambiguity
# (e.g. the 2-operand pseudo "BZ reg, label" below vs. the 3-operand real
# hardware "BZ test_reg, base_reg, label").
#
# Structure:
#     PSEUDO_INSTRUCTION_SPEC = {
#         "name": {
#             "operands": [{"name": "reg1", "type": "LcrIdx"}, ...],
#             "expands_to": {
#                 "slot": "cond",
#                 "instruction": "BLT",
#                 "args": ["reg2", "reg1", "label"],
#                     # Real instruction's operands, in order. Each entry is
#                     # either a name from this pseudo's own "operands" list
#                     # (substituted with the caller's actual token), or a
#                     # literal register name such as "CR0" (assumed to
#                     # always hold zero, used to derive BZ/BNZ from
#                     # BEQ/BNE).
#             },
#             "doc": InstructionDoc(...),
#         },
#         ...
#     }

PSEUDO_INSTRUCTION_SPEC: dict[str, dict] = {
    "BGT": {
        "operands": [
            {"name": "reg1", "type": "LcrIdx"},
            {"name": "reg2", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BLT",
            "args": ["reg2", "reg1", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Greater Than (pseudo)",
            summary="Branch if first register is greater than second.",
            syntax="BGT reg1, reg2, label",
            operands=[
                "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
                "label: Branch target label",
            ],
            operation="if (reg1 > reg2) PC = label",
            example="BGT LR0, LR1, bigger;;",
            notes=(
                "Expands to `BLT reg2, reg1, label` at assemble time "
                "(operands swapped). Identical encoding and runtime cost "
                "to a hand-written BLT."
            ),
        ),
    },
    "BLE": {
        "operands": [
            {"name": "reg1", "type": "LcrIdx"},
            {"name": "reg2", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BGE",
            "args": ["reg2", "reg1", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Less or Equal (pseudo)",
            summary="Branch if first register is less than or equal to second.",
            syntax="BLE reg1, reg2, label",
            operands=[
                "reg1: First register to compare (LR0–LR15 or CR0–CR14)",
                "reg2: Second register to compare (LR0–LR15 or CR0–CR14)",
                "label: Branch target label",
            ],
            operation="if (reg1 <= reg2) PC = label",
            example="BLE LR0, LR1, smaller_or_equal;;",
            notes=(
                "Expands to `BGE reg2, reg1, label` at assemble time "
                "(operands swapped), exactly mirroring how BGT expands to "
                "BLT. Identical encoding and runtime cost to a hand-written "
                "BGE."
            ),
        ),
    },
    "BZ": {
        "operands": [
            {"name": "reg", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BEQ",
            "args": ["reg", "CR0", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Zero (pseudo)",
            summary="Branch if register is zero. Assumes CR0 always holds zero.",
            syntax="BZ reg, label",
            operands=[
                "reg: Register to test (LR0–LR15 or CR0–CR14)",
                "label: Branch target label",
            ],
            operation="if (reg == 0) PC = label",
            example="BZ LR0, done;;",
            notes="Expands to `BEQ reg, CR0, label`. Assumes CR0 always holds 0.",
        ),
    },
    "BNZ": {
        "operands": [
            {"name": "reg", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BNE",
            "args": ["reg", "CR0", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Not Zero (pseudo)",
            summary="Branch if register is not zero. Assumes CR0 always holds zero.",
            syntax="BNZ reg, label",
            operands=[
                "reg: Register to test (LR0–LR15 or CR0–CR14)",
                "label: Branch target label",
            ],
            operation="if (reg != 0) PC = label",
            example="BNZ LR0, loop;;",
            notes="Expands to `BNE reg, CR0, label`. Assumes CR0 always holds 0.",
        ),
    },
    "B": {
        "operands": [
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BEQ",
            "args": ["CR0", "CR0", "label"],
        },
        "doc": InstructionDoc(
            title="Unconditional Branch (pseudo)",
            summary="Always branch to label.",
            syntax="B label",
            operands=["label: Branch target label"],
            operation="PC = label",
            example="B start;;",
            notes="Expands to `BEQ CR0, CR0, label`. CR0 always equals itself, so the branch is always taken.",
        ),
    },
}


def find_pseudo_instruction(name: str, arg_count: int) -> dict | None:
    """Look up a pseudo-instruction definition by name and operand count.

    Matching is case-insensitive on the name AND requires an exact operand
    count match, so a pseudo-instruction never shadows a real instruction
    of a different arity sharing the same name. Returns None if no
    pseudo-instruction matches.
    """
    lowered = name.lower()
    for pseudo_name, pseudo_def in PSEUDO_INSTRUCTION_SPEC.items():
        if pseudo_name.lower() == lowered and len(pseudo_def["operands"]) == arg_count:
            return pseudo_def
    return None


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


def validate_pseudo_instruction_spec() -> None:
    """Validate pseudo-instruction specification consistency.

    Checks:
    - Each pseudo-instruction has 'operands', 'expands_to', and 'doc'
    - No pseudo-instruction defines 'execute_fn' (they never reach the
      emulator — only the assembler expands them)
    - Operands have name/type fields with a recognized type
    - 'expands_to' references a real slot + instruction that actually
      exists in INSTRUCTION_SPEC, with a matching operand count
    - Each 'expands_to.args' entry is either one of this pseudo's own
      operand names or a literal string (e.g. "CR0")

    Raises ValueError if validation fails.
    """
    for name, pseudo_def in PSEUDO_INSTRUCTION_SPEC.items():
        if "operands" not in pseudo_def:
            raise ValueError(f"pseudo.{name}: missing 'operands' field")
        if "expands_to" not in pseudo_def:
            raise ValueError(f"pseudo.{name}: missing 'expands_to' field")
        if "doc" not in pseudo_def:
            raise ValueError(f"pseudo.{name}: missing 'doc' field")
        if "execute_fn" in pseudo_def:
            raise ValueError(
                f"pseudo.{name}: pseudo-instructions must not define "
                f"'execute_fn' — they expand to a real instruction at "
                f"assemble time and never reach the emulator"
            )

        operands = pseudo_def["operands"]
        if not isinstance(operands, list):
            raise ValueError(f"pseudo.{name}: 'operands' must be a list of dicts")

        operand_names = set()
        for operand in operands:
            if "name" not in operand or "type" not in operand:
                raise ValueError(
                    f"pseudo.{name}: each operand needs a 'name' and 'type'"
                )
            if operand["type"] not in VALID_OPERAND_TYPES:
                raise ValueError(
                    f"pseudo.{name}: operand '{operand['name']}' has invalid "
                    f"type '{operand['type']}'"
                )
            operand_names.add(operand["name"])

        if not isinstance(pseudo_def["doc"], InstructionDoc):
            raise ValueError(f"pseudo.{name}: 'doc' must be InstructionDoc instance")

        expansion = pseudo_def["expands_to"]
        if not isinstance(expansion, dict):
            raise ValueError(f"pseudo.{name}: 'expands_to' must be a dict")

        slot = expansion.get("slot")
        real_instruction = expansion.get("instruction")
        args = expansion.get("args")

        if slot not in INSTRUCTION_SPEC:
            raise ValueError(
                f"pseudo.{name}: expands_to.slot '{slot}' is not a valid slot"
            )
        try:
            real_def = get_instruction(slot, real_instruction)
        except KeyError:
            raise ValueError(
                f"pseudo.{name}: expands_to.instruction '{real_instruction}' "
                f"not found in slot '{slot}'"
            )

        if not isinstance(args, list):
            raise ValueError(f"pseudo.{name}: 'expands_to.args' must be a list")
        if len(args) != len(real_def["operands"]):
            raise ValueError(
                f"pseudo.{name}: expands_to.args has {len(args)} entries but "
                f"{slot}.{real_instruction} takes {len(real_def['operands'])} operands"
            )
        for arg in args:
            if not isinstance(arg, str):
                raise ValueError(
                    f"pseudo.{name}: expands_to.args entries must be strings"
                )


# Validate on import
validate_instruction_spec()
validate_pseudo_instruction_spec()

# Derive union layout from INSTRUCTION_SPEC (must run after spec is fully defined).
from ipu_common.union_layout import compute_slot_layouts, SlotUnion  # noqa: E402

_computed_unions = compute_slot_layouts(INSTRUCTION_SPEC)
SLOT_UNIONS.update(_computed_unions)
SLOT_BINARY_LAYOUT.update({
    slot: [f.canonical_type for f in su.fields]
    for slot, su in _computed_unions.items()
})
