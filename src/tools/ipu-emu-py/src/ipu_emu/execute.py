"""VLIW dispatch loop and per-slot instruction executors.

This module is the Python equivalent of ipu.c + all ipu_*_inst.c files.
Each executor reads operand fields from a decoded instruction dict,
reads register values from a *snapshot* (VLIW read-before-write), and
writes results to the *live* state.

Instruction format
~~~~~~~~~~~~~~~~~~
An instruction is stored as a ``dict[str, int]`` whose keys match the
C struct field names produced by ``CompoundInst.get_fields()``, e.g.::

    {
        "break_inst_token_0_break_inst_opcode": 2,   # break_nop
        "xmem_inst_token_0_xmem_inst_opcode": 4,     # xmem_nop
        ...
    }

These dicts are produced by :func:`decode_instruction_word`.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import Any

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE
from ipu_emu.regfile import RegFile
from ipu_emu.ipu_math import ipu_mult, ipu_add, DType

# Re-export for convenience
from ipu_as.compound_inst import CompoundInst
from ipu_as import inst as inst_module

# ---------------------------------------------------------------------------
# Opcode index constants — derived from the enum_array() order in opcodes.py.
# The C code uses INST_PARSER__<TYPE>_OPCODE_<NAME> = <index>.
# ---------------------------------------------------------------------------

# XmemInstOpcode: ["str_acc_reg", "ldr_mult_reg", "ldr_cyclic_mult_reg", "ldr_mult_mask_reg", "xmem_nop"]
XMEM_OP_STR_ACC_REG = 0
XMEM_OP_LDR_MULT_REG = 1
XMEM_OP_LDR_CYCLIC_MULT_REG = 2
XMEM_OP_LDR_MULT_MASK_REG = 3
XMEM_OP_XMEM_NOP = 4

# LrInstOpcode: ["incr", "set", "add", "sub"]
LR_OP_INCR = 0
LR_OP_SET = 1
LR_OP_ADD = 2
LR_OP_SUB = 3

# MultInstOpcode: ["mult.ee", "mult.ev", "mult.ve", "mult_nop"]
MULT_OP_EE = 0
MULT_OP_EV = 1
MULT_OP_VE = 2
MULT_OP_NOP = 3

# AccInstOpcode: ["acc", "reset_acc", "acc_nop"]
ACC_OP_ACC = 0
ACC_OP_RESET = 1
ACC_OP_NOP = 2

# CondInstOpcode: ["beq", "bne", "blt", "bnz", "bz", "b", "br", "bkpt"]
COND_OP_BEQ = 0
COND_OP_BNE = 1
COND_OP_BLT = 2
COND_OP_BNZ = 3
COND_OP_BZ = 4
COND_OP_B = 5
COND_OP_BR = 6
COND_OP_BKPT = 7

# BreakInstOpcode: ["break", "break.ifeq", "break_nop"]
BREAK_OP_BREAK = 0
BREAK_OP_IFEQ = 1
BREAK_OP_NOP = 2

# MultStageRegField: ["r0", "r1", "mem_bypass"]
MULT_REG_R0 = 0
MULT_REG_R1 = 1
MULT_REG_MEM_BYPASS = 2

# Number of scalar (LR) registers — LCR indices >= 16 refer to CR registers.
LR_REG_COUNT = 16

# R register size in bytes
R_REG_SIZE = 128
R_CYCLIC_SIZE = 512
R_ACC_SIZE = 512


class BreakResult(IntEnum):
    CONTINUE = 0
    BREAK = 1


# ---------------------------------------------------------------------------
# Instruction word decoder
# ---------------------------------------------------------------------------

def decode_instruction_word(word: int) -> dict[str, int]:
    """Decode a raw integer instruction word into a field dict.

    Uses ``CompoundInst.get_fields()`` which returns ``[(name, bits), ...]``
    ordered from MSB to LSB.  We iterate in reverse (LSB first) to extract
    each field.
    """
    fields = CompoundInst.get_fields()  # LSB-first list
    result: dict[str, int] = {}
    shift = 0
    for name, width in fields:
        mask = (1 << width) - 1
        result[name] = (word >> shift) & mask
        shift += width
    return result


def _instruction_aligned_bytes() -> int:
    """Number of bytes per instruction in binary format (32-bit word aligned).

    Must match ``instruction_aligned_bytes_len()`` in the assembler.
    """
    bits = CompoundInst.bits()
    word_size = 32
    if bits % word_size != 0:
        bits += word_size - (bits % word_size)
    return (bits // word_size) * 4


def load_binary_instructions(data: bytes) -> list[dict[str, int]]:
    """Load a binary file (output of ``ipu-as assemble --format bin``) into
    a list of decoded instruction dicts.

    Each instruction is word-aligned to 32-bit boundaries (matching the
    assembler's ``assemble_to_bin_file``).
    """
    inst_bytes = _instruction_aligned_bytes()
    instructions: list[dict[str, int]] = []
    for offset in range(0, len(data), inst_bytes):
        chunk = data[offset : offset + inst_bytes]
        if len(chunk) < inst_bytes:
            break
        word = int.from_bytes(chunk, byteorder="little")
        instructions.append(decode_instruction_word(word))
    return instructions


# ---------------------------------------------------------------------------
# Helper: read LCR register (LR if index < 16, else CR[index - 16])
# ---------------------------------------------------------------------------

def _read_lcr(snapshot: RegFile, idx: int) -> int:
    if idx < LR_REG_COUNT:
        return snapshot.get_lr(idx)
    else:
        return snapshot.get_cr(idx - LR_REG_COUNT)


# ---------------------------------------------------------------------------
# Helper: get mult-stage register bytes
# ---------------------------------------------------------------------------

def _get_mult_stage_reg(state: IpuState, mult_stage_idx: int) -> bytearray:
    """Return 128 bytes of the selected mult-stage register."""
    if mult_stage_idx == MULT_REG_R0:
        return state.regfile.get_r(0)
    elif mult_stage_idx == MULT_REG_R1:
        return state.regfile.get_r(1)
    elif mult_stage_idx == MULT_REG_MEM_BYPASS:
        return state.regfile.get_mem_bypass()
    else:
        raise ValueError(f"Invalid mult_stage_idx: {mult_stage_idx}")


# ---------------------------------------------------------------------------
# BREAK executor
# ---------------------------------------------------------------------------

def execute_break(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> BreakResult:
    opcode = inst["break_inst_token_0_break_inst_opcode"]

    if opcode == BREAK_OP_BREAK:
        return BreakResult.BREAK

    if opcode == BREAK_OP_IFEQ:
        lr_idx = inst["break_inst_token_1_lr_reg_field"]
        lr_val = snapshot.get_lr(lr_idx)
        imm = inst["break_inst_token_2_break_immediate_type"]
        if lr_val == imm:
            return BreakResult.BREAK
        return BreakResult.CONTINUE

    # break_nop or default
    return BreakResult.CONTINUE


# ---------------------------------------------------------------------------
# XMEM executor
# ---------------------------------------------------------------------------

def execute_xmem(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    opcode = inst["xmem_inst_token_0_xmem_inst_opcode"]

    if opcode == XMEM_OP_XMEM_NOP:
        return

    lr_idx = inst["xmem_inst_token_2_lr_reg_field"]
    cr_idx = inst["xmem_inst_token_4_cr_reg_field"]
    lr_val = snapshot.get_lr(lr_idx)
    cr_val = snapshot.get_cr(cr_idx)
    addr = lr_val + cr_val

    if opcode == XMEM_OP_STR_ACC_REG:
        # Store accumulator to XMEM
        acc_data = state.regfile.get_r_acc_bytes()
        state.xmem.write_address(addr, acc_data)

    elif opcode == XMEM_OP_LDR_MULT_REG:
        data = state.xmem.read_address(addr, R_REG_SIZE)
        mult_stage_idx = inst["xmem_inst_token_1_mult_stage_reg_field"]
        if mult_stage_idx == MULT_REG_R0:
            state.regfile.set_r(0, data)
        elif mult_stage_idx == MULT_REG_R1:
            state.regfile.set_r(1, data)
        elif mult_stage_idx == MULT_REG_MEM_BYPASS:
            state.regfile.set_mem_bypass(data)

    elif opcode == XMEM_OP_LDR_CYCLIC_MULT_REG:
        data = state.xmem.read_address(addr, R_REG_SIZE)
        lr_idx_idx = inst["xmem_inst_token_3_lr_reg_field"]
        lr_idx_val = snapshot.get_lr(lr_idx_idx)
        assert lr_idx_val % R_REG_SIZE == 0, (
            f"LR index for cyclic load must be aligned to {R_REG_SIZE}: got {lr_idx_val}"
        )
        state.regfile.set_r_cyclic_at(lr_idx_val, data)

    elif opcode == XMEM_OP_LDR_MULT_MASK_REG:
        data = state.xmem.read_address(addr, R_REG_SIZE)
        state.regfile.set_r_mask(data)


# ---------------------------------------------------------------------------
# LR executor (2 slots in the compound instruction)
# ---------------------------------------------------------------------------

def _execute_single_lr(state: IpuState, opcode: int, lr_idx: int,
                       immediate: int, lcr_a: int, lcr_b: int,
                       snapshot: RegFile) -> None:
    """Execute one LR sub-instruction."""
    if opcode == LR_OP_SET:
        state.regfile.set_lr(lr_idx, immediate)

    elif opcode == LR_OP_INCR:
        old = snapshot.get_lr(lr_idx)
        state.regfile.set_lr(lr_idx, (old + immediate) & 0xFFFFFFFF)

    elif opcode == LR_OP_ADD:
        val_a = _read_lcr(snapshot, lcr_a)
        val_b = _read_lcr(snapshot, lcr_b)
        state.regfile.set_lr(lr_idx, (val_a + val_b) & 0xFFFFFFFF)

    elif opcode == LR_OP_SUB:
        val_a = _read_lcr(snapshot, lcr_a)
        val_b = _read_lcr(snapshot, lcr_b)
        state.regfile.set_lr(lr_idx, (val_a - val_b) & 0xFFFFFFFF)


def _is_lr_valid(opcode: int, immediate: int) -> bool:
    """INCR by 0 is effectively a NOP."""
    return not (opcode == LR_OP_INCR and immediate == 0)


def execute_lr(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    """Execute both LR slots, with conflict detection."""
    slots: list[tuple[int, int, int, int, int]] = []  # (opcode, lr_idx, imm, lcr_a, lcr_b)

    for slot in range(2):
        prefix = f"lr_inst_{slot}"
        opcode = inst[f"{prefix}_token_0_lr_inst_opcode"]
        lr_idx = inst[f"{prefix}_token_1_lr_reg_field"]

        if opcode in (LR_OP_ADD, LR_OP_SUB):
            lcr_a = inst[f"{prefix}_token_2_lcr_reg_field"]
            lcr_b = inst[f"{prefix}_token_3_lcr_reg_field"]
            imm = 0
        else:
            imm = inst[f"{prefix}_token_4_lr_immediate_type"]
            lcr_a = 0
            lcr_b = 0

        if _is_lr_valid(opcode, imm):
            slots.append((opcode, lr_idx, imm, lcr_a, lcr_b))

    # Conflict check: no two valid instructions may write to the same LR
    lr_targets = [s[1] for s in slots]
    if len(lr_targets) != len(set(lr_targets)):
        raise RuntimeError(
            f"LR conflict: multiple writes to LR{lr_targets} in same cycle"
        )

    for opcode, lr_idx, imm, lcr_a, lcr_b in slots:
        _execute_single_lr(state, opcode, lr_idx, imm, lcr_a, lcr_b, snapshot)


# ---------------------------------------------------------------------------
# MULT executor
# ---------------------------------------------------------------------------

def _mult_mask_and_shift(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    """Apply mask-and-shift to mult_res, zeroing masked-out positions."""
    lr_mask_idx_reg = inst["mult_inst_token_3_lr_reg_field"]
    lr_shift_reg = inst["mult_inst_token_4_lr_reg_field"]

    lr_mask_idx = snapshot.get_lr(lr_mask_idx_reg)
    lr_shift = snapshot.get_lr(lr_shift_reg)
    # Interpret lr_shift as signed int32
    if lr_shift >= 0x80000000:
        lr_shift = lr_shift - 0x100000000

    # The mask register is 128 bytes = 1024 bits.
    # It is accessed as an array of __uint128_t masks (8 masks of 128 bits each).
    # lr_mask_idx selects which 128-bit mask to use.
    mask_bytes = state.regfile.get_r_mask()
    mask_slot = lr_mask_idx % (R_REG_SIZE // 16)  # 128/16 = 8 slots of 128-bit
    # Extract 128-bit mask from the mask register
    offset = mask_slot * 16
    mask_int = int.from_bytes(mask_bytes[offset:offset + 16], byteorder="little")

    # Apply shift
    if lr_shift > 0:
        mask_int <<= lr_shift
    elif lr_shift < 0:
        mask_int >>= -lr_shift

    # Zero out mult_res where mask bit is set
    mult_res = state.regfile.raw("mult_res")
    for i in range(R_REG_SIZE):
        if (mask_int >> i) & 1:
            # Zero the uint32 word at position i
            struct.pack_into("<I", mult_res, i * 4, 0)


def execute_mult(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    opcode = inst["mult_inst_token_0_mult_inst_opcode"]

    if opcode == MULT_OP_NOP:
        return

    mult_stage_idx = inst["mult_inst_token_1_mult_stage_reg_field"]
    ra = _get_mult_stage_reg(state, mult_stage_idx)

    lr_cyclic_reg = inst["mult_inst_token_2_lr_reg_field"]
    lr_cyclic_val = snapshot.get_lr(lr_cyclic_reg)

    dtype = state.get_cr_dtype()

    # Get the cyclic register slice
    rb = state.regfile.get_r_cyclic_at(lr_cyclic_val, R_REG_SIZE)

    # mult_res is stored as 128 × uint32 words (512 bytes)
    mult_res = state.regfile.raw("mult_res")

    if opcode == MULT_OP_EE:
        # Element-wise: mult_res[i] = Ra[i] * Rb[i]
        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

    elif opcode == MULT_OP_EV:
        # Element × fixed cyclic: mult_res[i] = Ra[i] * Rb[0]
        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[0], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

    elif opcode == MULT_OP_VE:
        # Fixed Ra × cyclic element: mult_res[i] = Ra[fixed] * Rb[i]
        lr_fixed_ra_reg = inst["mult_inst_token_5_lr_reg_field"]
        lr_fixed_ra_val = snapshot.get_lr(lr_fixed_ra_reg)
        ra_fixed = ra[lr_fixed_ra_val % R_REG_SIZE]
        for i in range(R_REG_SIZE):
            result = ipu_mult(ra_fixed, rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

    _mult_mask_and_shift(state, inst, snapshot)


# ---------------------------------------------------------------------------
# ACC executor
# ---------------------------------------------------------------------------

def execute_acc(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    opcode = inst["acc_inst_token_0_acc_inst_opcode"]

    if opcode == ACC_OP_NOP:
        return

    if opcode == ACC_OP_RESET:
        state.regfile.set_r_acc_bytes(bytearray(R_ACC_SIZE))
        return

    if opcode == ACC_OP_ACC:
        dtype = state.get_cr_dtype()
        # Accumulate: acc[i] += mult_res[i]  (word-by-word, 128 words)
        acc_buf = state.regfile.raw("r_acc")
        mult_res = state.regfile.raw("mult_res")
        snap_acc = snapshot.raw("r_acc")
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = ipu_add(acc_val, mult_val, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)


# ---------------------------------------------------------------------------
# COND executor
# ---------------------------------------------------------------------------

def execute_cond(state: IpuState, inst: dict[str, int], snapshot: RegFile) -> None:
    opcode = inst["cond_inst_token_0_cond_inst_opcode"]
    lr1_idx = inst["cond_inst_token_1_lr_reg_field"]
    lr2_idx = inst["cond_inst_token_2_lr_reg_field"]
    lr1 = snapshot.get_lr(lr1_idx)
    lr2 = snapshot.get_lr(lr2_idx)
    label = inst["cond_inst_token_3_label_token"]

    if opcode == COND_OP_BEQ:
        state.program_counter = label if lr1 == lr2 else state.program_counter + 1
    elif opcode == COND_OP_BNE:
        state.program_counter = label if lr1 != lr2 else state.program_counter + 1
    elif opcode == COND_OP_BLT:
        state.program_counter = label if lr1 < lr2 else state.program_counter + 1
    elif opcode == COND_OP_BNZ:
        state.program_counter = label if lr1 != 0 else state.program_counter + 1
    elif opcode == COND_OP_BZ:
        state.program_counter = label if lr1 == 0 else state.program_counter + 1
    elif opcode == COND_OP_B:
        state.program_counter = label
    elif opcode == COND_OP_BR:
        state.program_counter = lr1
    elif opcode == COND_OP_BKPT:
        state.program_counter = INST_MEM_SIZE  # halt


# ---------------------------------------------------------------------------
# Top-level VLIW dispatch
# ---------------------------------------------------------------------------

def execute_next_instruction(state: IpuState) -> BreakResult:
    """Execute one VLIW cycle.

    1. Fetch instruction at ``state.program_counter``
    2. Snapshot the register file
    3. Execute BREAK first (before side effects)
    4. Execute XMEM, LR, MULT, ACC, COND in parallel from the snapshot
    """
    inst = state.inst_mem[state.program_counter]
    if inst is None:
        # NOP — just advance PC
        state.program_counter += 1
        return BreakResult.CONTINUE

    snapshot = state.regfile.snapshot()

    # Break runs first — may halt before side effects
    result = execute_break(state, inst, snapshot)
    if result == BreakResult.BREAK:
        return BreakResult.BREAK

    # Execute all other slots using the snapshot
    execute_xmem(state, inst, snapshot)
    execute_lr(state, inst, snapshot)
    execute_mult(state, inst, snapshot)
    execute_acc(state, inst, snapshot)
    execute_cond(state, inst, snapshot)

    return BreakResult.CONTINUE


def execute_instruction_skip_break(state: IpuState) -> None:
    """Execute the current instruction without re-checking break.

    Used after returning from a debug break to complete the cycle.
    """
    inst = state.inst_mem[state.program_counter]
    if inst is None:
        state.program_counter += 1
        return

    snapshot = state.regfile.snapshot()
    execute_xmem(state, inst, snapshot)
    execute_lr(state, inst, snapshot)
    execute_mult(state, inst, snapshot)
    execute_acc(state, inst, snapshot)
    execute_cond(state, inst, snapshot)
