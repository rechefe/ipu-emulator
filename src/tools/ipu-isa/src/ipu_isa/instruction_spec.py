"""Master instruction specification for IPU — single source of truth.

This module defines ALL IPU instructions in one place with complete metadata:
- Instruction name and operands
- Operand definitions (name, type, meaning)
- Documentation (syntax, examples)
- Execution handler reference (for emulator)

KEY DESIGN PRINCIPLES:
1. NO EXPLICIT OPCODES: Opcode is derived from instruction position within slot.
   Position 0 → opcode 0, position 1 → opcode 1, etc.

2. DATA ONLY: this package imports nothing from the assembler or the
   emulator. Each consumer derives what it needs from the dicts here (the
   assembler builds its own opcode token classes).

3. STRUCTURED OPERANDS: Each operand has a meaningful name and type.
   Types are string names the assembler resolves to its token classes.
   Operands with a ``"read"`` field are source registers whose values
   are auto-resolved by the emulator dispatcher before calling handlers.
   The value is the timing view (``ipu_isa.timing``, GLOSSARY.md):
     - ``"cycle_start"`` → the value the register held when the word began
     - ``"forwarded"``   → the value after the writes of earlier slots of
       the same word
   Operands with ``"write": True`` are destination registers: the raw index
   is passed through and the handler writes it.

4. TIMING: every register an instruction reads or writes is declared, as an
   operand (``read`` / ``write``) or in ``implicit_reads`` /
   ``implicit_writes``. ``ipu_isa.timing`` holds the slot execution order and
   the timing contract the declarations are checked against.

OPERAND TYPE NAMES (resolved by the assembler into its token classes):
  - "MultStageReg": R0 or R1 (MultStageRegField); 2-bit encoding in the VLIW word
  - "LrIdx": LR0–LR15 (LrRegField)
  - "CrIdx": CR0–CR15 (CrRegField)
  - "LcrIdx": LR0–LR15 or CR0–CR15 (LcrRegField)
  - "LrdIdx": LRD0, LRD2, LRD4, ..., LRD14 register-pair alias over LR (LrdRegField), named after the lower register in the pair; no physical storage — LRDn = LR(n+1):LR(n)
  - "LrIncDecImmediate": unsigned immediate for INC/DEC; bit width derived from LR slot union layout
  - "AddbiImmediate": byte immediate for ADDBI (0–255, or an equivalent signed literal); reinterpreted as signed int8 for the add
  - "LrModPow2KImmediate": k operand for INC.MODP2 (semantic k ∈ [1, 9]; encoded as k−1 in 4 bits)
  - "MultMaskOffsetImmediate": mask slot index for mult masking (0–7; eight 128-bit slots in R_MASK)
  - "LrOrReshapeMaskImmediate": element-start index for RESHAPE; selects trailing elements (reshape_mask..7); accepts immediate 0–7 or an LR register (bit width, and therefore the range of encodable LR registers, derived from union layout)
  - "ActivationFn": keyword on `ACTIVATE.QUANTIZE` (see ``ACTIVATION_FN_NAMES`` in ``activations.py``)
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
                    {"name": "src", "type": "LrIdx", "read": "cycle_start"},
                    {"name": "dest", "type": "LrIdx", "write": True},
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
        - ``"read": "cycle_start"`` → source register resolved to the value it
          held when the instruction word began.
        - ``"read": "forwarded"`` → source register resolved to the value
          after the writes of earlier slots in the same word.
        - Absent → destination/index operand; the raw index is passed
          through so the handler can write to it (``"write": True`` when it
          is the register the instruction writes).
        - Immediates and Labels have no "read" flag (literal values always
          passed as-is).

    Instruction "implicit_reads" / "implicit_writes" (optional): registers
    the handler reads or writes that are not ``read`` / ``write`` operands, as
    ``{"register": <REGISTER_DEFINITIONS name>, "via_operand": <operand that
    selects the index, optional>, "view": "cycle_start" | "forwarded"}``
    (``view`` on reads only).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ipu_isa import schema
from ipu_isa.constants import CR_ONE_REG_INDEX, CR_ZERO_REG_INDEX
from ipu_isa.mnemonics import check_mnemonic
from ipu_isa.operand_types import OPERAND_TYPES, OperandKind, OperandType
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.slots import LAYOUT_ORDER, SLOTS, Slot, SlotInfo
from ipu_isa.tags import Tag

# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "InstructionDoc",
    "syntax",
    "INSTRUCTION_SPEC",
    "SLOT_COUNT",
    "SLOT_METADATA",
    "COMPOUND_LAYOUT_SLOT_ORDER",
    "VALID_OPERAND_TYPES",
    "validate_instruction_spec",
    "PSEUDO_INSTRUCTION_SPEC",
    "validate_pseudo_instruction_spec",
]


# ===========================================================================
# Documentation Type
# ===========================================================================


@dataclass(frozen=True)
class InstructionDoc:
    """Documentation for an instruction (used by assembler for help/docs).

    There is no syntax line: :func:`syntax` derives it from the mnemonic and the
    operand names, so it cannot disagree with them. ``example`` is real code,
    with operands separated by spaces as the grammar requires. Frozen; the
    authored ``operands`` list is stored as a tuple.
    """

    title: str
    summary: str
    operands: Sequence[str]  # authored as a list, stored as a tuple
    operation: str | None = None
    example: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))


def syntax(mnemonic: str, operand_names: Sequence[str]) -> str:
    """The syntax line of an instruction: ``MNEMONIC operand operand ...``."""
    return " ".join([mnemonic, *operand_names])


# The binary layout (each slot's union fields, word widths) is derived from
# INSTRUCTION_SPEC by ipu_isa.view.load_isa (isa().layout). Slots are
# tagged HARDWARE or EMULATOR in ipu_isa.slots; the generated SV package renders
# the HARDWARE slots only (E03-22), the emulator binary all of them.

# How many times each slot appears in the VLIW instruction word.
# Most slots appear once; LR appears three times (three independent sub-instructions).
# Derived views over the slot registry (ipu_isa.slots), kept so callers can
# migrate to SLOTS one at a time.

#: Instances of each slot in one word.
SLOT_COUNT: dict[str, int] = {slot.value: info.count for slot, info in SLOTS.items()}

#: Compound instruction binary layout (MSB → LSB).
COMPOUND_LAYOUT_SLOT_ORDER: list[str] = [slot.value for slot in LAYOUT_ORDER]

#: Per-slot description.
SLOT_METADATA: dict[str, dict[str, Any]] = {
    slot.value: {"description": info.description} for slot, info in SLOTS.items()
}


def _instruction_tag(
    slot: str, inst_def: dict[str, Any], slots: Mapping[Slot, SlotInfo] = SLOTS
) -> Tag:
    """An instruction's tag: its own ``"tag"`` if it has one, else its slot's.

    The schema requires the two to agree, so this is always the slot's tag.
    """
    tag: Tag = inst_def.get("tag", slots[Slot(slot)].tag)
    return tag


def _pseudo_tag(pseudo_def: dict[str, Any]) -> Tag:
    """A pseudo-instruction's tag: ``ASSEMBLER`` (the schema rejects any other)."""
    tag: Tag = pseudo_def.get("tag", Tag.ASSEMBLER)
    return tag


# ===========================================================================
# MASTER INSTRUCTION SPECIFICATION
# ===========================================================================
# Each slot type (load, store, acc_store, lr, mult, acc, cond, break) is defined
# separately.  Instructions maintain ORDER — position in dict determines opcode!
# ===========================================================================

INSTRUCTION_SPEC = {
    # =========================================================================
    # LOAD Slot (first pipeline stage — feeds the multiply unit)
    # Opcode = position in list: LD.MULT=0, LD.MULT.CYCLIC=1, etc.
    # =========================================================================
    "load": {
        "LD.MULT": {
            "operands": [
                {"name": "dest", "type": "MultStageReg", "write": True},
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
                {"name": "base", "type": "CrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Load Register",
                summary="Load data from memory into a multiplication stage register.",
                operands=[
                    "`dest`: **`R0`** | **`R1`** — mult-stage register to load (2-bit field; only these encodings are valid).",
                    "`offset`: **`LR0`**…**`LR15`** — offset register, in XMEM rows.",
                    "`base`: **`CR0`**…**`CR15`** — base address register, in XMEM rows.",
                ],
                operation="dest = Memory[row(offset + base)]  # one row = 128 elements (128 B narrow, 512 B wide-vector debug)",
                example="SET LR0 CR1;;\nLD.MULT R0 LR0 CR0;;",
            ),
            "execute_fn": "execute_ld_mult",
        },
        "LD.MULT.CYCLIC": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
                {"name": "base", "type": "CrIdx", "read": "cycle_start"},
                {"name": "index", "type": "LrIdx", "read": "forwarded"},
            ],
            "doc": InstructionDoc(
                title="Load Cyclic Register",
                summary="Load with cyclic addressing into R_CYCLIC.",
                operands=[
                    "offset: Offset register (LR0–LR15), in XMEM rows.",
                    "base: Base address register (CR0–CR15), in XMEM rows.",
                    "index: Index inside cyclic register (LR0–LR15); must hold 0, 128, 256, or 384 — the four R_CYCLIC slot boundaries, counted in **elements**. Any other value raises an error. All four are valid in **both** modes: R_CYCLIC is a 512-element ring in narrow and wide-vector debug mode alike, and a load always replaces one whole 128-element slot.",
                ],
                operation="R_CYCLIC[index .. index+127] = Memory[row(offset + base)]   # index ∈ {0, 128, 256, 384}",
            ),
            "implicit_writes": [{"register": "r_cyclic"}],
            "execute_fn": "execute_ld_mult_cyclic",
        },
        "LD.MASK": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
                {"name": "base", "type": "CrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Load Mask Register",
                summary="Load mask data from memory.",
                operands=[
                    "offset: Offset register (LR0–LR15), in XMEM rows.",
                    "base: Base address register (CR0–CR15), in XMEM rows.",
                ],
                operation="R_MASK = Memory[row(offset + base) .. +128B]  # reads only the row's first 128 B, in both modes -- the mask is 1 bit/element and does not scale with element width",
            ),
            "implicit_writes": [{"register": "r_mask"}],
            "execute_fn": "execute_ld_mask",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for load slot.",
                operands=[],
            ),
            "execute_fn": "execute_load_nop",
        },
    },
    # =========================================================================
    # STORE Slot (last pipeline stage — drains POST_AAQ_REG after AAQ/activate)
    # =========================================================================
    "store": {
        "ST.AAQ": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
                {"name": "base", "type": "CrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Store Post-AAQ register",
                summary=(
                    "Write the whole **`POST_AAQ_REG`** to one XMEM row. **Interim:** after "
                    "`ACTIVATE.QUANTIZE` it holds 128 INT8 results followed by zeros, until "
                    "quantization export is finalized."
                ),
                operands=[
                    "offset: Offset register (LR0–LR15), in XMEM rows.",
                    "base: Base address register (CR0–CR15), in XMEM rows.",
                ],
                operation="Memory[row(offset + base)] = POST_AAQ_REG (the whole register, one row, in every element format)",
                example="ST.AAQ LR0 CR0;;",
            ),
            "implicit_reads": [
                {"register": "post_aaq_reg", "view": "forwarded"},
            ],
            "execute_fn": "execute_st_aaq",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for store slot.",
                operands=[],
            ),
            "execute_fn": "execute_store_nop",
        },
    },
    # =========================================================================
    # ACC_STORE Slot (simulation-only — NOT implemented in real IPU hardware)
    # =========================================================================
    "acc_store": {
        "ST.ACC": {
            "operands": [
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
                {"name": "base", "type": "CrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Store Accumulator",
                summary=(
                    "Store **R_ACC** to external memory. **Simulation-only** — not "
                    "implemented in real IPU hardware."
                ),
                operands=[
                    "offset: Offset register (LR0–LR15), in XMEM rows.",
                    "base: Base address register (CR0–CR15), in XMEM rows.",
                ],
                operation="Memory[row(offset + base)] = R_ACC (128 32-bit elements, in every element format)",
                example="ST.ACC LR0 CR1;;",
                notes="This instruction lives in the simulation-only **acc_store** slot.",
            ),
            "implicit_reads": [
                {"register": "r_acc", "view": "forwarded"},
            ],
            "execute_fn": "execute_st_acc",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for acc_store slot (simulation-only).",
                operands=[],
            ),
            "execute_fn": "execute_acc_store_nop",
        },
    },
    # =========================================================================
    # LR Slot (Loop Register Instructions)
    # Opcode = position: SET=0, ADD=1, SUB=2, INC.MODP2=3, INC=4, DEC=5,
    #          ADDB=6, ADDBI=7, NOP=8
    # =========================================================================
    "lr": {
        "SET": {
            "operands": [
                {"name": "reg", "type": "LrIdx", "write": True},
                {"name": "src", "type": "CrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Set Loop Register",
                summary="Copy a 32-bit value from a configuration register into a loop register.",
                operands=[
                    "reg: Loop register (LR0–LR15)",
                    "src: Source configuration register (CR0–CR15)",
                ],
                operation="reg = cr[src]",
                example="SET LR0 CR1;;",
            ),
            "execute_fn": "execute_lr_set",
        },
        "ADD": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "src_a", "type": "LrIdx", "read": "cycle_start"},
                {"name": "src_b", "type": "LcrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Add",
                summary=("Add two register sources and store the result in the destination LR."),
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15 or CR0–CR15",
                ],
                operation="dest = src_a + src_b",
                example="ADD LR0 LR1 LR2;;\nADD LR3 LR1 CR5;;",
            ),
            "execute_fn": "execute_lr_add",
        },
        "SUB": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "src_a", "type": "LrIdx", "read": "cycle_start"},
                {"name": "src_b", "type": "LcrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Subtract",
                summary=(
                    "Subtract the second source from the first and store the result "
                    "in the destination LR."
                ),
                operands=[
                    "dest: Destination local register (LR0–LR15)",
                    "src_a: First source local register (LR0–LR15)",
                    "src_b: Second source — LR0–LR15 or CR0–CR15",
                ],
                operation="dest = src_a - src_b",
                example="SUB LR0 LR1 LR2;;\nSUB LR3 LR1 CR5;;",
            ),
            "execute_fn": "execute_lr_sub",
        },
        "INC.MODP2": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "step", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "k", "type": "LrModPow2KImmediate"},
            ],
            "doc": InstructionDoc(
                title="Increment Loop Register Modulo Power of Two",
                summary=(
                    "Add a loop or configuration register into the destination loop "
                    "register, then mask to k low bits (mod 2^k)."
                ),
                operands=[
                    "dest: Destination loop register (LR0–LR15); read and written",
                    "step: Signed 32-bit increment from LR0–LR15 or CR0–CR15",
                    "k: Immediate in [1, 9]; encoded in 4 bits as (k − 1); mask = (1 << k) - 1",
                ],
                operation="dest <- (dest + step) & ((1 << k) - 1)",
                example="INC.MODP2 LR2 LR3 4;;",
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "dest", "view": "cycle_start"},
            ],
            "execute_fn": "execute_lr_inc_modp2",
        },
        "INC": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "imm", "type": "LrIncDecImmediate"},
            ],
            "doc": InstructionDoc(
                title="Increment",
                summary=("Add an unsigned immediate to the destination LR (read-modify-write)."),
                operands=[
                    "dest: Destination local register (LR0–LR15); also the implicit source",
                    "imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout",
                ],
                operation="dest = dest + imm",
                example="INC LR0 7;;",
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "dest", "view": "cycle_start"},
            ],
            "execute_fn": "execute_lr_inc",
        },
        "DEC": {
            "operands": [
                {"name": "dest", "type": "LrIdx", "write": True},
                {"name": "imm", "type": "LrIncDecImmediate"},
            ],
            "doc": InstructionDoc(
                title="Decrement",
                summary=(
                    "Subtract an unsigned immediate from the destination LR (read-modify-write)."
                ),
                operands=[
                    "dest: Destination local register (LR0–LR15); also the implicit source",
                    "imm: Unsigned immediate; range 0 to 2^W − 1 where W is derived from the LR slot union layout",
                ],
                operation="dest = dest - imm",
                example="DEC LR0 3;;",
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "dest", "view": "cycle_start"},
            ],
            "execute_fn": "execute_lr_dec",
        },
        "ADDB": {
            "operands": [
                {"name": "dest", "type": "LrdIdx", "write": True},
                {"name": "src_b", "type": "LcrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Add Byte (Broadcast, Register)",
                summary=(
                    "Broadcast-add a signed byte from an LR/CR register's low byte to each of "
                    "the 8 byte elements of an LRDn register pair, saturating each element to [0, 255]."
                ),
                operands=[
                    "dest: Destination register pair (LRD0, LRD2, LRD4, ..., LRD14 — named after the lower register); LRDn = LR(n+1):LR(n), 8 byte elements; also the implicit source",
                    "src_b: LR0–LR15 or CR0–CR15; low byte is read and reinterpreted as a signed two's-complement byte",
                ],
                operation=(
                    "byte_val = src_b & 0xFF  # interpreted as signed int8 (two's complement)\n"
                    "for i in 0..7: dest_byte[i] = clamp(dest_byte[i] + byte_val, 0, 255)"
                ),
                example="ADDB LRD0 LR3;;\nADDB LRD2 CR5;;",
                notes=(
                    "There is no separate subtract opcode: since src_b's byte is reinterpreted as "
                    "signed, a value ≥ 128 (i.e. negative in two's complement) subtracts from every "
                    "element instead of adding. E.g. an LR/CR whose low byte is 200 (== -56) decreases "
                    "each element by 56. Because elements saturate at both ends of [0, 255], subtracting "
                    "past 0 clamps at 0 rather than wrapping — it does not behave like a plain "
                    "two's-complement subtract."
                ),
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "dest", "view": "cycle_start"},
            ],
            "execute_fn": "execute_addb",
        },
        "ADDBI": {
            "operands": [
                {"name": "dest", "type": "LrdIdx", "write": True},
                {"name": "imm", "type": "AddbiImmediate"},
            ],
            "doc": InstructionDoc(
                title="Add Byte Immediate (Broadcast)",
                summary=(
                    "Broadcast-add a signed immediate byte to each of the 8 byte elements of an "
                    "LRDn register pair, saturating each element to [0, 255]."
                ),
                operands=[
                    "dest: Destination register pair (LRD0, LRD2, LRD4, ..., LRD14 — named after the lower register); LRDn = LR(n+1):LR(n), 8 byte elements; also the implicit source",
                    "imm: Unsigned byte 0–255 (or an equivalent signed −128–127 literal); reinterpreted as a signed two's-complement byte for the add",
                ],
                operation=(
                    "byte_val = imm  # interpreted as signed int8 (two's complement)\n"
                    "for i in 0..7: dest_byte[i] = clamp(dest_byte[i] + byte_val, 0, 255)"
                ),
                example="ADDBI LRD0 200;;\nADDBI LRD2 7;;",
                notes=(
                    "There is no separate subtract opcode: write imm as a negative literal (e.g. "
                    "-56) or its equivalent unsigned encoding (200) to subtract from every element "
                    "instead of adding — both spellings encode the same bit pattern. Because elements "
                    "saturate at both ends of [0, 255], subtracting past 0 clamps at 0 rather than "
                    "wrapping — it does not behave like a plain two's-complement subtract."
                ),
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "dest", "view": "cycle_start"},
            ],
            "execute_fn": "execute_addbi",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for LR slot.",
                operands=[],
            ),
            "execute_fn": "execute_lr_nop",
        },
    },
    # =========================================================================
    # MULT Slot (Multiply Instructions)
    # Opcode = position: MULT.RC.VV=0, MULT.RC.VE=1, MULT.RC.VS=2, NOP=3,
    #          MULT.VE=4, MULT.EE=5
    # =========================================================================
    "mult": {
        "MULT.RC.VV": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "forwarded"},
                {"name": "ra", "type": "MultStageReg", "read": "cycle_start"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "forwarded"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="RC Vector × Ra Vector Multiply",
                summary=(
                    "Multiply R_CYCLIC[rc_idx:rc_idx+128] by Ra (R0 or R1) element-wise. "
                    "The partition used by mask_shift comes from cr_idx's dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`).",
                    "`ra`: **`R0`** | **`R1`** — multiplicand mult-stage register (same cycle as `LD.MULT` into **`R0`**/**`R1`** is allowed).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                    "`cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly).",
                ],
                operation="For each element i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], ra[i]); then apply mask and shift using CR[cr_idx].partition.",
                example="MULT.RC.VV LR0 R0 0 LR2 CR15;;",
                notes="Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "implicit_reads": [
                {"register": "r_cyclic", "view": "cycle_start"},
                {"register": "r_mask", "view": "forwarded"},
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "mult_res"}],
            "execute_fn": "execute_mult_rc_vv",
        },
        "MULT.RC.VE": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "forwarded"},
                {"name": "src", "type": "LcrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "forwarded"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="RC Vector × Scalar Multiply",
                summary=(
                    "Multiply R_CYCLIC[rc_idx:rc_idx+128] by a scalar, element-wise. The scalar is "
                    "either a single element of R0/R1 (selected by an LR value) or a CR register value. "
                    "The partition used by mask_shift comes from cr_idx's dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`).",
                    "`src`: **`LR0`**…**`LR15`** | **`CR0`**…**`CR15`** — if an LR, its stored value selects the scalar from R0/R1 (0..127 → `R0[idx]`, 128..255 → `R1[idx - 128]`); if a CR, its low byte supplies the scalar directly.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                    "`cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly).",
                ],
                operation="For each element i: MULT_RES[i] = ipu_mult(R_CYCLIC[(rc_idx + i) % 512], src_value); then apply mask and shift using CR[cr_idx].partition.",
                example="MULT.RC.VE LR0 LR3 0 LR2 CR15;;",
                notes="Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "src", "view": "forwarded"},
                {"register": "cr", "via_operand": "src", "view": "cycle_start"},
                {"register": "r", "view": "cycle_start"},
                {"register": "r_cyclic", "view": "cycle_start"},
                {"register": "r_mask", "view": "forwarded"},
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "mult_res"}],
            "execute_fn": "execute_mult_rc_ve",
        },
        "MULT.RC.VS": {
            "operands": [
                {"name": "rc_idx", "type": "LrIdx", "read": "forwarded"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "forwarded"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="RC Vector Self-Multiply (Square)",
                summary=(
                    "Square R_CYCLIC[rc_idx:rc_idx+128] element-wise. "
                    "The partition used by mask_shift comes from cr_idx's dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "`rc_idx`: **`LR0`**…**`LR15`** — base ELEMENT index into **`R_CYCLIC`** (cyclic, mod 512 elements; same unit as `LD.MULT.CYCLIC`'s `index`).",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                    "`cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly).",
                ],
                operation="For each element i: rb = R_CYCLIC[(rc_idx + i) % 512]; MULT_RES[i] = ipu_mult(rb, rb); then apply mask and shift using CR[cr_idx].partition.",
                example="MULT.RC.VS LR0 0 LR2 CR15;;",
                notes="Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "implicit_reads": [
                {"register": "r_cyclic", "view": "cycle_start"},
                {"register": "r_mask", "view": "forwarded"},
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "mult_res"}],
            "execute_fn": "execute_mult_rc_vs",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for multiply slot.",
                operands=[],
            ),
            "execute_fn": "execute_mult_nop",
        },
        "MULT.VE": {
            "operands": [
                {"name": "ra_idx", "type": "LrIdx", "read": "forwarded"},
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "forwarded"},
                {"name": "dstructure_cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Ra Vector × CR Scalar Multiply",
                summary=(
                    "Multiply Ra[ra_idx:ra_idx+128] (combined R0/R1) by a CR scalar, element-wise. "
                    "The partition used by mask_shift comes from dstructure_cr_idx's dstructure "
                    "configuration (any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "`ra_idx`: **`LR0`**…**`LR15`** — base element offset into combined Ra (`R0` ++ `R1`, 256 elements, cyclic mod 256).",
                    "`cr_idx`: **`CR0`**…**`CR15`** — CR register whose low byte supplies the scalar multiplier.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                    "`dstructure_cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly).",
                ],
                operation="For each element i: ra = Ra[(ra_idx + i) % 256]; MULT_RES[i] = ipu_mult(ra, CR[cr_idx][0]); then apply mask and shift using CR[dstructure_cr_idx].partition.",
                example="MULT.VE LR0 CR3 0 LR2 CR15;;",
                notes="Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "r", "view": "cycle_start"},
                {"register": "r_mask", "view": "forwarded"},
                {"register": "cr", "via_operand": "dstructure_cr_idx", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "mult_res"}],
            "execute_fn": "execute_mult_ve",
        },
        "MULT.EE": {
            "operands": [
                {"name": "ra_idx", "type": "LrIdx", "read": "forwarded"},
                {"name": "cr_idx", "type": "CrIdx"},
                {"name": "mask_offset", "type": "MultMaskOffsetImmediate"},
                {"name": "mask_shift", "type": "LrIdx", "read": "forwarded"},
                {"name": "dstructure_cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Ra Element × CR Scalar Multiply (broadcast)",
                summary=(
                    "Multiply a single Ra element by a CR scalar; broadcast the product to all 128 elements. "
                    "The partition used by mask_shift comes from dstructure_cr_idx's dstructure "
                    "configuration (any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "`ra_idx`: **`LR0`**…**`LR15`** — index of the single Ra element to read (combined `R0` ++ `R1`, 256 elements, mod 256).",
                    "`cr_idx`: **`CR0`**…**`CR15`** — CR register whose low byte supplies the scalar multiplier.",
                    "`mask_offset`: immediate mask slot **`0`**…**`7`** — selects one of eight 128-bit masks in **`R_MASK`**.",
                    "`mask_shift`: **`LR0`**…**`LR15`** — index ∈ [−3, +3] (values >3 clamp to 3, values <−3 clamp to −3) selecting one of seven masks via sequential shift-and-AND: positive indices use partition_vector (0 at group start), negative indices use inverse_partition_vector (0 at group end).",
                    "`dstructure_cr_idx`: **`CR0`**…**`CR15`** — dstructure register supplying the `partition` field used to build the partition vectors and the `pad_mode` field used to fill deactivated elements (must be given explicitly).",
                ],
                operation="ra = Ra[ra_idx % 256]; result = ipu_mult(ra, CR[cr_idx][0]); for each element i: MULT_RES[i] = result; then apply mask and shift using CR[dstructure_cr_idx].partition.",
                example="MULT.EE LR0 CR3 0 LR2 CR15;;",
                notes="Element masking via `mask_offset` and `mask_shift` fills elements whose derived mask bit is **0** (deactivated) in `MULT_RES` with the dstructure register's `pad_mode` value (zero by default, or +inf/-inf for a floating-point dtype) before accumulation; elements with bit **1** pass through. See [Masking](assembly-syntax.md#masking) for the full algorithm.",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "r", "view": "cycle_start"},
                {"register": "r_mask", "view": "forwarded"},
                {"register": "cr", "via_operand": "dstructure_cr_idx", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "mult_res"}],
            "execute_fn": "execute_mult_ee",
        },
    },
    # =========================================================================
    # ACC Slot (Accumulator Instructions)
    # Opcode = position: ACC.ADD=0, ACC.ADD.FIRST=1, ACC.MAX=2, ACC.MAX.FIRST=3,
    #          ACC.SUB=4, ACC.SUB.FIRST=5, NOP=6, ACC.STRIDE=7,
    #          AGG.SUM.FIRST=8, AGG.SUM=9, AGG.MAX.FIRST=10, AGG.MAX=11
    # =========================================================================
    "acc": {
        "ACC.ADD": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Add",
                summary="Running add accumulation: add multiply result into each R_ACC element.",
                operands=[],
                operation="R_ACC[i] += MULT_RES[i]  # for each element i",
                example="ACC.ADD;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
                {"register": "r_acc", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_add",
        },
        "ACC.ADD.FIRST": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Add (First)",
                summary="Clean-init add: overwrite each R_ACC element with the multiply result (ignores previous R_ACC).",
                operands=[],
                operation="R_ACC[i] = MULT_RES[i]  # for each element i",
                example="ACC.ADD.FIRST;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_add_first",
        },
        "ACC.MAX": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Max",
                summary="Running max accumulation: each R_ACC element takes the max of its current value and the multiply result.",
                operands=[],
                operation="R_ACC[i] = max(R_ACC[i], MULT_RES[i])  # for each element i",
                example="ACC.MAX;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
                {"register": "r_acc", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_max",
        },
        "ACC.MAX.FIRST": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Max (First)",
                summary="Clean-init max: overwrite each R_ACC element unconditionally with the multiply result.",
                operands=[],
                operation="R_ACC[i] = MULT_RES[i]  # for each element i (unconditional overwrite)",
                example="ACC.MAX.FIRST;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_max_first",
        },
        "ACC.SUB": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Subtract",
                summary="Running subtract accumulation: subtract the multiply result from each R_ACC element each cycle.",
                operands=[],
                operation="R_ACC[i] -= MULT_RES[i]  # for each element i",
                example="ACC.SUB;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
                {"register": "r_acc", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_sub",
        },
        "ACC.SUB.FIRST": {
            "operands": [],
            "doc": InstructionDoc(
                title="Accumulate Subtract (First)",
                summary="Clean-init subtract: set each R_ACC element to the negated multiply result.",
                operands=[],
                operation="R_ACC[i] = -MULT_RES[i]  # for each element i",
                example="ACC.SUB.FIRST;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_sub_first",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for accumulator slot.",
                operands=[],
            ),
            "execute_fn": "execute_acc_nop",
        },
        "ACC.STRIDE": {
            "operands": [
                {"name": "elements_in_row", "type": "ElementsInRow"},
                {"name": "horizontal_stride", "type": "HorizontalStride"},
                {"name": "vertical_stride", "type": "VerticalStride"},
                {"name": "offset", "type": "LrIdx", "read": "forwarded"},
            ],
            "doc": InstructionDoc(
                title="Accumulator Stride",
                summary="Reorder the multiplication result into R_ACC using horizontal/vertical stride decimation. Only updates the RACC indexes written; leaves the rest unchanged.",
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
                example="ACC.STRIDE 16 off off LR0;;",
            ),
            "implicit_reads": [
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_stride",
        },
        "AGG.SUM.FIRST": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "cycle_start"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Sum (First)",
                summary=(
                    "Sum active MULT_RES elements and write the result into R_ACC at the slot given by LR. "
                    "The current value at the destination slot is NOT included in the sum (clean initialisation). "
                    "Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly)",
                ],
                operation=(
                    "Let n = min(CR[cr_idx].valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = sum(MULT_RES[0..n-1])."
                ),
                example="AGG.SUM.FIRST LR0 CR3;;",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_agg_sum_first",
        },
        "AGG.SUM": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "cycle_start"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Sum",
                summary=(
                    "Sum active MULT_RES elements and ADD the result to R_ACC at the slot given by LR "
                    "(running cross-cycle accumulation). "
                    "Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly)",
                ],
                operation=(
                    "Let n = min(CR[cr_idx].valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = sum(MULT_RES[0..n-1]) + R_ACC[dest]."
                ),
                example="AGG.SUM LR0 CR3;;",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "mult_res", "view": "forwarded"},
                {"register": "r_acc", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_agg_sum",
        },
        "AGG.MAX.FIRST": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "cycle_start"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Max (First)",
                summary=(
                    "Find the maximum of active MULT_RES elements and write it into R_ACC at the slot given by LR. "
                    "The current value at the destination slot is NOT used as a seed (clean initialisation). "
                    "Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly)",
                ],
                operation=(
                    "Let n = min(CR[cr_idx].valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = max(MULT_RES[0..n-1]); when n = 0 the identity seed "
                    "(INT32_MIN for integer elements, -inf for float elements) is written."
                ),
                example="AGG.MAX.FIRST LR0 CR3;;",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "mult_res", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_agg_max_first",
        },
        "AGG.MAX": {
            "operands": [
                {"name": "dest_slot", "type": "LrIdx", "read": "cycle_start"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Aggregate Max",
                summary=(
                    "Find the maximum of active MULT_RES elements seeded with the current destination slot value "
                    "(running cross-cycle max). "
                    "Only the active element count (``valid_elements``) is read from ``cr_idx``'s dstructure configuration "
                    "(any CR0-CR15; must be named explicitly)."
                ),
                operands=[
                    "dest_slot: LR register whose value gives the destination slot in R_ACC (0–127)",
                    "cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly)",
                ],
                operation=(
                    "Let n = min(CR[cr_idx].valid_elements, 128). "
                    "dest = LR[dest_slot] % 128. "
                    "R_ACC[dest] = max(MULT_RES[0..n-1], R_ACC[dest])."
                ),
                example="AGG.MAX LR0 CR3;;",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "mult_res", "view": "forwarded"},
                {"register": "r_acc", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_agg_max",
        },
        "ACC.RESHAPE": {
            "operands": [
                {"name": "source", "type": "LrdIdx", "read": "cycle_start"},
                {"name": "dest", "type": "LrdIdx", "read": "cycle_start"},
                {"name": "reshape_mask", "type": "LrOrReshapeMaskImmediate"},
            ],
            "doc": InstructionDoc(
                title="Reshape",
                summary=(
                    "Scatter MULT_RES elements into R_ACC: for each of the trailing "
                    "(8 - reshape_mask) byte elements of the source/dest LRDn pairs "
                    "(indices reshape_mask..7), copy MULT_RES[source[i]] to R_ACC[dest[i]]."
                ),
                operands=[
                    "source: LRDn register pair (LRD0, LRD2, ..., LRD14) read as source[0..7] — 8 byte elements, each a MULT_RES word index",
                    "dest: LRDn register pair (LRD0, LRD2, ..., LRD14) read as dest[0..7] — 8 byte elements, each an R_ACC word index",
                    "reshape_mask: immediate 0-7 or an LR register; selects the first participating element index (elements reshape_mask..7 participate; reshape_mask = 0 uses all 8, reshape_mask = 7 uses only element 7). When an LR is given, its full value is used as the mask at runtime.",
                ],
                operation=(
                    "mult_res = MULT_RES (cycle-start value)\n"
                    "mask = reshape_mask if immediate else LR[reshape_mask]\n"
                    "if mask > 8: raise error\n"
                    "for i in mask..7:\n"
                    "    if source[i] >= 128 or dest[i] >= 128: raise error\n"
                    "    R_ACC[dest[i]] = mult_res[source[i]]"
                ),
                example="ACC.RESHAPE LRD0 LRD2 0;;",
                notes=(
                    "reshape_mask skips the lowest-indexed elements 0..reshape_mask-1; the "
                    "rest participate. Example: reshape_mask = 2 skips elements 0, 1; "
                    "elements 2-7 participate. Example: reshape_mask = 5 skips elements "
                    "0-4; elements 5-7 participate."
                ),
            ),
            "implicit_reads": [
                {"register": "lr", "via_operand": "reshape_mask", "view": "forwarded"},
                {"register": "mult_res", "view": "cycle_start"},
            ],
            "implicit_writes": [{"register": "r_acc"}],
            "execute_fn": "execute_acc_reshape",
        },
    },
    # =========================================================================
    # AAQ Slot (Activation and Quantization)
    # Opcode = position: NOP=0, ACTIVATE.QUANTIZE=1
    # =========================================================================
    "aaq": {
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for AAQ slot.",
                operands=[],
            ),
            "execute_fn": "execute_aaq_nop",
        },
        "ACTIVATE.QUANTIZE": {
            "operands": [
                {"name": "activation_fn", "type": "ActivationFn"},
                {"name": "cr_idx", "type": "DstructureCrIdx"},
            ],
            "doc": InstructionDoc(
                title="Activate and Quantize",
                summary=(
                    "Read active elements from the forwarded R_ACC (this cycle's ACC result), apply the selected element-wise activation, "
                    "clamp the results to the INT8 range ``[-128, 127]``, and write the resulting bytes to "
                    "the leading active-element positions of **``POST_AAQ_REG``**; all remaining bytes are zeroed. "
                    "``r_acc`` is not modified. Requires INT8 mode. "
                    "The active element count is taken from ``cr_idx``'s dstructure ``valid_elements`` field "
                    "(any CR0–CR15; must be named explicitly). "
                    "Available activation functions: ``identity`` (0), ``relu`` (1), ``relu6`` (2), "
                    "``sigmoid`` (3), ``tanh`` (4), ``gelu`` (5), ``softplus`` (6), ``elu`` (7), "
                    "``exp2`` (8), ``reciprocal`` (9), ``rsqrt`` (10), ``silu`` (11), ``window`` (12)."
                ),
                operands=[
                    "activation_fn: keyword naming the activation (one of identity, relu, relu6, sigmoid, tanh, gelu, softplus, elu, exp2, reciprocal, rsqrt, silu, window; see ACTIVATION_FN_NAMES)",
                    "cr_idx: CR0…CR15 supplying valid_elements (must be given explicitly)",
                ],
                operation=(
                    "Requires the INT8 element format (IpuState.fmt == ElementFormat.INT8). "
                    "Let n = min(CR[cr_idx].valid_elements, 128) and k = encoded activation index. "
                    "For i in [0, n): POST_AAQ_REG[i] = clamp(round(activation_k(R_ACC[i])), -128, 127). "
                    "POST_AAQ_REG[n..511] = 0. R_ACC is not modified. "
                    "Clamping is a placeholder until a full per-element requantize is implemented. "
                    "The selector uses four bits; encodings outside the thirteen named activations behave as identity. "
                    "window(x) = 1 for a <= x < b and 0 otherwise. "
                    "α for elu and the window bounds a, b are not ISA operands; see "
                    "docs/content/building-applications.md#activations-emulator."
                ),
                example="ACTIVATE.QUANTIZE relu CR15;;",
            ),
            "implicit_reads": [
                {"register": "cr", "via_operand": "cr_idx", "view": "cycle_start"},
                {"register": "r_acc", "view": "forwarded"},
            ],
            "implicit_writes": [{"register": "post_aaq_reg"}],
            "execute_fn": "execute_activate_quantize",
        },
    },
    # =========================================================================
    # COND Slot (Conditional Branch Instructions)
    # Opcode = position: BEQ=0, BNE=1, BLT.S=2, BGE.S=3, BR=4, HALT=5, NOP=6
    # BNZ, BZ, and B are pseudo-instructions (see PSEUDO_INSTRUCTION_SPEC):
    # they're each exactly expressible via BEQ/BNE against CR0 (always zero),
    # so they don't need their own opcode.
    # =========================================================================
    "cond": {
        "BEQ": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "reg2", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Equal",
                summary="Branch if two registers are equal.",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 == reg2) PC = label",
                example="BEQ LR0 LR1 end;;\nend:\nHALT;;",
            ),
            "execute_fn": "execute_beq",
        },
        "BNE": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "reg2", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Not Equal",
                summary="Branch if two registers are not equal.",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (reg1 != reg2) PC = label",
                example="loop:\nINC LR0 1;;\nBNE LR0 CR0 loop;;",
            ),
            "execute_fn": "execute_bne",
        },
        "BLT.S": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "reg2", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Less Than (signed)",
                summary="Branch if first register is less than second, as signed 32-bit integers.",
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (int32(reg1) < int32(reg2)) PC = label  # signed 32-bit",
                example="BLT.S LR0 CR1 smaller;;\nsmaller:\nHALT;;",
            ),
            "execute_fn": "execute_blt_s",
        },
        "BGE.S": {
            "operands": [
                {"name": "reg1", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "reg2", "type": "LcrIdx", "read": "cycle_start"},
                {"name": "label", "type": "Label"},
            ],
            "doc": InstructionDoc(
                title="Branch if Greater or Equal (signed)",
                summary=(
                    "Branch if first register is greater than or equal to second, "
                    "as signed 32-bit integers."
                ),
                operands=[
                    "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                    "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                    "label: Branch target label",
                ],
                operation="if (int32(reg1) >= int32(reg2)) PC = label  # signed 32-bit",
                example="BGE.S LR0 CR1 ge;;\nge:\nHALT;;",
            ),
            "execute_fn": "execute_bge_s",
        },
        "BR": {
            "operands": [
                {"name": "reg", "type": "LcrIdx", "read": "cycle_start"},
            ],
            "doc": InstructionDoc(
                title="Branch Register",
                summary="Branch to address in register.",
                operands=["reg: Register containing target address (LR0–LR15 or CR0–CR15)"],
                operation="PC = reg",
            ),
            "execute_fn": "execute_br",
        },
        "HALT": {
            "operands": [],
            "doc": InstructionDoc(
                title="Halt",
                summary="End the kernel after every slot of this word has retired.",
                operands=[],
                operation="no further fetch; kernel done when this word retires",
                example="HALT;;",
                notes=(
                    "Unconditional. The other slots of the same word retire first; then the "
                    "kernel is done. To stop for debugging, use **`BREAK`** (emulator-only) "
                    "instead. Not the host's `HALT` command, which stops a running kernel."
                ),
            ),
            "execute_fn": "execute_halt",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for COND slot; advances PC by one.",
                operands=[],
            ),
            "execute_fn": "execute_cond_nop",
        },
    },
    # =========================================================================
    # BREAK Slot (Break Instructions)
    # Opcode = position: BREAK=0, BREAK.IFEQ=1, NOP=2
    # =========================================================================
    "break": {
        "BREAK": {
            "operands": [],
            "doc": InstructionDoc(
                title="Break",
                summary="Unconditional break.",
                operands=[],
                operation="Halt execution",
            ),
            "execute_fn": "execute_break",
        },
        "BREAK.IFEQ": {
            "operands": [
                {"name": "reg", "type": "LrIdx", "read": "cycle_start"},
                {"name": "value", "type": "BreakImmediate"},
            ],
            "doc": InstructionDoc(
                title="Break if Equal",
                summary="Break execution if register equals value.",
                operands=[
                    "reg: Register to test (LR0–LR15)",
                    "value: Immediate value to compare against",
                ],
                operation="if (reg == value) BREAK",
                example="BREAK.IFEQ LR0 10;;",
            ),
            "execute_fn": "execute_break_ifeq",
        },
        "NOP": {
            "operands": [],
            "doc": InstructionDoc(
                title="No Operation",
                summary="No operation for BREAK slot.",
                operands=[],
            ),
            "execute_fn": "execute_break_nop",
        },
    },
}


# ===========================================================================
# Operand types (single source for validation + documentation generation)
# ===========================================================================

#: Every operand-type name, from the operand-type registry.
VALID_OPERAND_TYPES: frozenset[str] = frozenset(OPERAND_TYPES)


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
#                 "instruction": "BLT.S",
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

PSEUDO_INSTRUCTION_SPEC: dict[str, dict[str, Any]] = {
    "BGT.S": {
        "operands": [
            {"name": "reg1", "type": "LcrIdx"},
            {"name": "reg2", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BLT.S",
            "args": ["reg2", "reg1", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Greater Than (pseudo)",
            summary="Branch if first register is greater than second.",
            operands=[
                "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                "label: Branch target label",
            ],
            operation="if (reg1 > reg2) PC = label",
            example="BGT.S LR0 LR1 bigger;;\nbigger:\nHALT;;",
            notes=(
                "Expands to `BLT.S reg2 reg1 label` at assemble time "
                "(operands swapped). Identical encoding and runtime cost "
                "to a hand-written BLT.S."
            ),
        ),
    },
    "BLE.S": {
        "operands": [
            {"name": "reg1", "type": "LcrIdx"},
            {"name": "reg2", "type": "LcrIdx"},
            {"name": "label", "type": "Label"},
        ],
        "expands_to": {
            "slot": "cond",
            "instruction": "BGE.S",
            "args": ["reg2", "reg1", "label"],
        },
        "doc": InstructionDoc(
            title="Branch if Less or Equal (pseudo)",
            summary="Branch if first register is less than or equal to second.",
            operands=[
                "reg1: First register to compare (LR0–LR15 or CR0–CR15)",
                "reg2: Second register to compare (LR0–LR15 or CR0–CR15)",
                "label: Branch target label",
            ],
            operation="if (reg1 <= reg2) PC = label",
            example="BLE.S LR0 LR1 smaller_or_equal;;\nsmaller_or_equal:\nHALT;;",
            notes=(
                "Expands to `BGE.S reg2 reg1 label` at assemble time "
                "(operands swapped), exactly mirroring how BGT.S expands to "
                "BLT.S. Identical encoding and runtime cost to a hand-written "
                "BGE.S."
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
            summary="Branch if register is zero (CR0 is the constant 0).",
            operands=[
                "reg: Register to test (LR0–LR15 or CR0–CR15)",
                "label: Branch target label",
            ],
            operation="if (reg == 0) PC = label",
            example="BZ LR0 done;;\ndone:\nHALT;;",
            notes="Expands to `BEQ reg CR0 label`; CR0 is the read-only constant 0.",
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
            summary="Branch if register is not zero (CR0 is the constant 0).",
            operands=[
                "reg: Register to test (LR0–LR15 or CR0–CR15)",
                "label: Branch target label",
            ],
            operation="if (reg != 0) PC = label",
            example="loop:\nDEC LR0 1;;\nBNZ LR0 loop;;",
            notes="Expands to `BNE reg CR0 label`; CR0 is the read-only constant 0.",
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
            operands=["label: Branch target label"],
            operation="PC = label",
            example="start:\nINC LR0 1;;\nB start;;",
            notes="Expands to `BEQ CR0 CR0 label`. CR0 always equals itself, so the branch is always taken.",
        ),
    },
}


# ===========================================================================
# Validation
# ===========================================================================

_OPERAND_NAME = re.compile(r"[a-z][a-z0-9_]*")
_DOC_OPERAND_NAME = re.compile(r"\s*`?([A-Za-z_][A-Za-z0-9_]*)`?\s*:")

#: Operand types a ``read`` key is allowed on: the readable register types.
READABLE_OPERAND_TYPES = frozenset(
    name for name, operand_type in OPERAND_TYPES.items() if operand_type.readable
)


def _check_operands(path: str, operands: object, operand_types: Mapping[str, OperandType]) -> None:
    """Keys, name, type and ``read`` of each operand; names unique."""
    readable = sorted(name for name, record in operand_types.items() if record.readable)
    if not isinstance(operands, list):
        raise schema.SpecError(f"{path}.operands: must be a list of dicts")
    for index, operand in enumerate(operands):
        name = operand.get("name", index) if isinstance(operand, dict) else index
        where = f"{path}.operands[{name}]"
        schema.check_keys(where, operand, schema.OPERAND)
        if not isinstance(operand["name"], str) or not _OPERAND_NAME.fullmatch(operand["name"]):
            raise schema.SpecError(
                f"{where}.name: {operand['name']!r} does not match [a-z][a-z0-9_]*"
            )
        if operand["type"] not in operand_types:
            raise schema.SpecError(
                f"{where}.type: unknown operand type {operand['type']!r}; "
                f"must be one of {sorted(operand_types)}"
            )
        if "read" in operand:
            if operand["read"] not in schema.READ_SOURCES:
                raise schema.SpecError(
                    f"{where}.read: {operand['read']!r} is not one of {sorted(schema.READ_SOURCES)}"
                )
            if operand["type"] not in readable:
                raise schema.SpecError(
                    f"{where}.read: a {operand['type']} operand is not a register, "
                    f"so it cannot be read; 'read' is allowed only on {readable}"
                )
        if "write" in operand:
            _check_write(where, operand, readable)
    schema.check_unique(path, "operand name", (op["name"] for op in operands))


def _check_write(where: str, operand: dict[str, Any], readable: list[str]) -> None:
    """``write`` is ``True`` on a register operand that has no ``read``: the
    handler gets the index to write, not a value."""
    if operand["write"] is not True:
        raise schema.SpecError(f"{where}.write: must be True, got {operand['write']!r}")
    if operand["type"] not in readable:
        raise schema.SpecError(
            f"{where}.write: a {operand['type']} operand is not a register, "
            f"so it cannot be written; 'write' is allowed only on {readable}"
        )
    if "read" in operand:
        raise schema.SpecError(
            f"{where}: a 'write' operand has no 'read' (the handler needs its index); "
            "declare the read in implicit_reads with via_operand"
        )


def _check_accesses(path: str, inst_def: dict[str, Any]) -> None:
    """``implicit_reads`` / ``implicit_writes`` entries name a register of
    REGISTER_DEFINITIONS, an operand of the instruction (``via_operand``) and,
    for a read, a view. The timing rules are checked by ``ipu_isa.timing``."""
    operand_names = {operand["name"] for operand in inst_def["operands"]}
    for key, keys in (
        ("implicit_reads", schema.IMPLICIT_READ),
        ("implicit_writes", schema.IMPLICIT_WRITE),
    ):
        if key not in inst_def:
            continue
        entries = inst_def[key]
        if not isinstance(entries, list) or not entries:
            raise schema.SpecError(f"{path}.{key}: must be a non-empty list of dicts")
        seen: set[tuple[str, str | None]] = set()
        for index, entry in enumerate(entries):
            where = f"{path}.{key}[{index}]"
            schema.check_keys(where, entry, keys)
            if entry["register"] not in REGISTER_DEFINITIONS:
                raise schema.SpecError(
                    f"{where}.register: {entry['register']!r} is not a register of "
                    "REGISTER_DEFINITIONS"
                )
            via = entry.get("via_operand")
            if via is not None and via not in operand_names:
                raise schema.SpecError(f"{where}.via_operand: {via!r} is not an operand")
            if "view" in entry and entry["view"] not in schema.READ_SOURCES:
                raise schema.SpecError(
                    f"{where}.view: {entry['view']!r} is not one of {sorted(schema.READ_SOURCES)}"
                )
            if (entry["register"], via) in seen:
                raise schema.SpecError(
                    f"{where}: {entry['register']!r} via {via!r} is declared twice"
                )
            seen.add((entry["register"], via))


def _check_doc(path: str, doc: object, operands: list[dict[str, Any]]) -> None:
    """``doc`` is an InstructionDoc whose operand lines name the operands, in order."""
    if not isinstance(doc, InstructionDoc):
        raise schema.SpecError(f"{path}.doc: must be an InstructionDoc")
    if doc.example is not None and "," in doc.example:
        raise schema.SpecError(
            f"{path}.doc.example: separate operands with spaces, not commas "
            "(the grammar has no comma)"
        )
    documented = []
    for line in doc.operands:
        match = _DOC_OPERAND_NAME.match(line)
        documented.append(match.group(1) if match else line)
    expected = [operand["name"] for operand in operands]
    if documented != expected:
        raise schema.SpecError(
            f"{path}.doc.operands: names {documented} do not match the operands {expected}"
        )


def _check_instruction_tag(
    path: str, slot_type: str, inst_def: dict[str, Any], slots: Mapping[Slot, SlotInfo]
) -> None:
    """An instruction has its slot's tag: an instruction in an EMULATOR slot is
    EMULATOR, and an EMULATOR instruction may not sit in a HARDWARE slot."""
    if "tag" not in inst_def:
        return
    tag, slot_tag = inst_def["tag"], slots[Slot(slot_type)].tag
    if not isinstance(tag, Tag):
        raise schema.SpecError(f"{path}.tag: must be a Tag, got {tag!r}")
    if tag is not slot_tag:
        raise schema.SpecError(
            f"{path}.tag: {tag.name} in a {slot_tag.name} slot; an instruction has its slot's tag"
        )


def validate_instruction_spec(
    instruction_spec: dict[str, Any],
    slot_metadata: dict[str, Any],
    *,
    slots: Mapping[Slot, SlotInfo] = SLOTS,
    operand_types: Mapping[str, OperandType] = OPERAND_TYPES,
) -> None:
    """Check the instruction spec and slot metadata against the schema, the
    slot registry ``slots`` and the operand-type registry ``operand_types``.

    A pure function of its arguments, so a test can check a modified copy.
    Raises :class:`ipu_isa.schema.SpecError` naming the path and the rule.
    """
    if set(instruction_spec) != {slot.value for slot in slots}:
        raise schema.SpecError(
            f"INSTRUCTION_SPEC: slots {sorted(instruction_spec)} do not match the slot "
            f"registry {sorted(slot.value for slot in slots)}"
        )
    if set(slot_metadata) != set(instruction_spec):
        raise schema.SpecError(
            f"SLOT_METADATA: slots {sorted(slot_metadata)} do not match "
            f"INSTRUCTION_SPEC slots {sorted(instruction_spec)}"
        )
    for slot_type, metadata in slot_metadata.items():
        schema.check_keys(f"SLOT_METADATA.{slot_type}", metadata, schema.SLOT_METADATA_ENTRY)

    for slot_type, instructions in instruction_spec.items():
        if not isinstance(instructions, dict) or not instructions:
            raise schema.SpecError(f"{slot_type}: must be a non-empty dict of instructions")
        schema.check_unique(
            slot_type, "mnemonic (compared in upper case)", (n.upper() for n in instructions)
        )
        nops = [name for name in instructions if name.upper() == "NOP"]
        if len(nops) != 1:
            raise schema.SpecError(f"{slot_type}: must have exactly one NOP, has {len(nops)}")
        for inst_name, inst_def in instructions.items():
            path = f"{slot_type}.{inst_name}"
            check_mnemonic(path, inst_name, Slot(slot_type))
            schema.check_keys(path, inst_def, schema.INSTRUCTION)
            _check_operands(path, inst_def["operands"], operand_types)
            _check_accesses(path, inst_def)
            _check_doc(path, inst_def["doc"], inst_def["operands"])
            if not isinstance(inst_def["execute_fn"], str):
                raise schema.SpecError(f"{path}.execute_fn: must be a string")
            _check_instruction_tag(path, slot_type, inst_def, slots)


_CR_LITERAL = re.compile(r"CR(\d+)", re.IGNORECASE)
_CONSTANT_CRS = (CR_ZERO_REG_INDEX, CR_ONE_REG_INDEX)


def _check_constant_register_literal(where: str, name: str, arg: str, doc: InstructionDoc) -> None:
    """A CR named literally in an expansion must be a constant register (CR0
    holds 0, CR1 holds 1), and the pseudo's docs must name it, so the
    expansion's meaning rests on a documented ISA fact."""
    match = _CR_LITERAL.fullmatch(arg)
    if match is None:
        return
    if int(match.group(1)) not in _CONSTANT_CRS:
        allowed = ", ".join(f"CR{index}" for index in _CONSTANT_CRS)
        raise schema.SpecError(f"{where}: literal {arg!r} is not a constant register ({allowed})")
    text = " ".join(filter(None, (doc.summary, doc.operation, doc.notes)))
    if not re.search(rf"\b{re.escape(arg)}\b", text, re.IGNORECASE):
        raise schema.SpecError(
            f"{where}: pseudo-instruction {name} expands with {arg!r} but its docs do not say so"
        )


def validate_pseudo_instruction_spec(
    pseudo_spec: dict[str, Any],
    instruction_spec: dict[str, Any],
    *,
    slots: Mapping[Slot, SlotInfo] = SLOTS,
    operand_types: Mapping[str, OperandType] = OPERAND_TYPES,
) -> None:
    """Check the pseudo-instruction spec against the schema and the real
    instructions it expands to.

    Each ``expands_to.args`` entry is one of the pseudo's own operand names or
    a literal register name in the target operand type's domain (``CR0`` for an
    ``LcrIdx``). Pseudo-instructions carry no ``execute_fn``: they expand at
    assemble time and never reach the emulator. A pure function of its
    arguments; raises :class:`ipu_isa.schema.SpecError`.
    """
    for name, pseudo_def in pseudo_spec.items():
        path = f"pseudo.{name}"
        check_mnemonic(path, name, None)
        schema.check_keys(path, pseudo_def, schema.PSEUDO_INSTRUCTION)
        _check_operands(path, pseudo_def["operands"], operand_types)
        _check_doc(path, pseudo_def["doc"], pseudo_def["operands"])
        if _pseudo_tag(pseudo_def) is not Tag.ASSEMBLER:
            raise schema.SpecError(f"{path}.tag: a pseudo-instruction is Tag.ASSEMBLER")
        _check_expansion(path, name, pseudo_def, instruction_spec, slots, operand_types)


def _check_expansion(  # noqa: PLR0913 -- the pseudo, its spec and the two registries
    path: str,
    name: str,
    pseudo_def: dict[str, Any],
    instruction_spec: dict[str, Any],
    slots: Mapping[Slot, SlotInfo],
    operand_types: Mapping[str, OperandType],
) -> None:
    """The pseudo's ``expands_to`` names a real instruction and fills each of
    its operands with a pseudo operand or a valid literal register."""
    expansion = pseudo_def["expands_to"]
    schema.check_keys(f"{path}.expands_to", expansion, schema.EXPANSION)
    slot, real_instruction, args = (
        expansion["slot"],
        expansion["instruction"],
        expansion["args"],
    )
    if slot not in instruction_spec:
        raise schema.SpecError(f"{path}.expands_to.slot: {slot!r} is not a slot")
    real_def = next(
        (
            d
            for n, d in instruction_spec[slot].items()
            if n.upper() == str(real_instruction).upper()
        ),
        None,
    )
    if real_def is None:
        raise schema.SpecError(
            f"{path}.expands_to.instruction: {real_instruction!r} is not in slot {slot!r}"
        )
    if _instruction_tag(slot, real_def, slots) not in (Tag.HARDWARE, Tag.EMULATOR):
        raise schema.SpecError(
            f"{path}.expands_to.instruction: {real_instruction!r} is not a HARDWARE "
            "or EMULATOR instruction"
        )
    targets = real_def["operands"]
    if not isinstance(args, list) or len(args) != len(targets):
        raise schema.SpecError(
            f"{path}.expands_to.args: must be a list of {len(targets)} entries, "
            f"one per operand of {slot}.{real_instruction}"
        )
    own_names = {operand["name"] for operand in pseudo_def["operands"]}
    for arg, target in zip(args, targets, strict=True):
        where = f"{path}.expands_to.args[{target['name']}]"
        if not isinstance(arg, str):
            raise schema.SpecError(f"{where}: must be a string")
        if arg in own_names:
            continue
        target_type = operand_types[target["type"]]
        registers = {member.upper() for member in target_type.domain.members}
        if target_type.kind is not OperandKind.REGISTER or arg.upper() not in registers:
            raise schema.SpecError(
                f"{where}: literal {arg!r} is neither an operand of {name} nor a "
                f"register valid for a {target['type']} operand"
            )
        _check_constant_register_literal(where, name, arg, pseudo_def["doc"])


# Validate on import
validate_instruction_spec(INSTRUCTION_SPEC, SLOT_METADATA)
validate_pseudo_instruction_spec(PSEUDO_INSTRUCTION_SPEC, INSTRUCTION_SPEC)
