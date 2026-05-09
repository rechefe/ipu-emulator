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
import warnings
from enum import IntEnum
from typing import Any

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE, WideVectorArithmetic
from ipu_emu.regfile import RegFile
from ipu_emu.ipu_math import ipu_mult, ipu_add, dtype_one_byte, DType
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
from ipu_common.acc_agg_enums import (
    AGG_MODE_SUM,
    AGG_MODE_MAX,
    POST_FN_VALUE,
    POST_FN_VALUE_CR,
    POST_FN_INV,
    POST_FN_INV_SQRT,
    get_agg_mode,
    get_post_fn,
)
from ipu_common.incr_mod_pow2_k import LR_MOD_POW2_K_ENCODED_MAX, LR_MOD_POW2_K_MIN
from ipu_common.registers import get_register_sizes, get_mult_stage_map

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EmulatorError(RuntimeError):
    """Raised when the emulator detects an invalid operation."""


# ---------------------------------------------------------------------------
# Constants — derived from the single source of truth in ipu-common
# ---------------------------------------------------------------------------

_emu_constants = create_emulator_constants()
_reg_sizes = get_register_sizes()

# MultStageRegField: index → (register_name, element_index)
# e.g. [("r", 0), ("r", 1)] — encoding 2 is reserved / invalid in assembly
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
    "AddSubSrcB": "add_sub_src_b_field",
    "AaqRegIdx": "aaq_reg_field",
    "ElementsInRow": "elements_in_row_field",
    "HorizontalStride": "horizontal_stride_field",
    "VerticalStride": "vertical_stride_field",
    "AggMode": "agg_mode_field",
    "PostFn": "post_fn_field",
    "Immediate": "lr_immediate_type",
    "LrModPow2KImmediate": "lr_mod_pow2_k_immediate",
    "BreakImmediate": "break_immediate_type",
    "Label": "label_token",
}

# Field prefix for each slot type (matches compound_inst naming)
_SLOT_FIELD_PREFIX = {
    "xmem": "xmem_inst",
    "mult": "mult_inst",
    "acc": "acc_inst",
    "aaq": "aaq_inst",
    "cond": "cond_inst",
    "break": "break_inst",
    # LR is special: "lr_inst_0", "lr_inst_1", ... (count from SLOT_COUNT["lr"])
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
      ("lr", inst_name, slot_idx)  — for LR sub-slots (0 .. SLOT_COUNT["lr"]-1)
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
    # Wide-vector debug mode (emulator-only; GitHub issue #33)
    # -----------------------------------------------------------------------

    def _wide_vector_active(self) -> bool:
        return self.state.wide_vector_debug

    def _wide_assert_lane_aligned_byte_offset(self, name: str, byte_off: int) -> None:
        """Wide-vector mode treats r_cyclic in 4-byte lanes; misaligned offsets corrupt unpacking."""
        if byte_off % 4 != 0:
            raise EmulatorError(
                f"Wide-vector debug: {name} must be 4-byte aligned, got {byte_off}"
            )

    def _wide_pack_aaq_bits(self, fmt: str, result_val: float | int) -> int:
        """Pack agg/aaq scalar result for ``set_aaq`` (uint32 bit pattern)."""
        if fmt == "<f":
            return struct.unpack("<I", struct.pack("<f", float(result_val)))[0]
        if isinstance(result_val, float):
            ri = int(round(result_val))
        else:
            ri = int(result_val)
        ri = max(-0x80000000, min(0x7FFFFFFF, ri))
        return ri & 0xFFFFFFFF

    def _wide_unpack_lane_tuple(self, buf: bytes | bytearray) -> tuple[float, ...] | tuple[int, ...]:
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            return struct.unpack_from("<128f", buf, 0)
        return struct.unpack_from("<128i", buf, 0)

    def _wide_imult32(self, a: int, b: int) -> int:
        p = int(a) * int(b)
        p &= 0xFFFFFFFF
        return p - 0x100000000 if p >= 0x80000000 else p

    def _wide_add_lane(self, a: float | int, b: float | int) -> float | int:
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            return float(a) + float(b)
        return ipu_add(int(a), int(b), DType.INT8)

    def _wide_aaq_scalar(self, aaq_rf_idx: int) -> float | int:
        raw = self.state.regfile.get_aaq(aaq_rf_idx) & 0xFFFFFFFF
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            return struct.unpack("<f", struct.pack("<I", raw))[0]
        return struct.unpack("<i", struct.pack("<I", raw))[0]

    def _wide_cr_scalar_byte_as_int32(self, cr_idx: int) -> int:
        """Low byte of CR as signed int32 lane (CR itself is not widened)."""
        b = self.state.regfile.get_cr(cr_idx) & 0xFF
        return b if b < 128 else b - 256

    def _debug_ra_lane_vals(self, mult_stage_enc: int) -> list[float | int]:
        snap = self.state._debug_mult_stage_vectors_snap
        if mult_stage_enc in snap:
            return list(snap[mult_stage_enc])
        return [0.0 if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32 else 0] * R_REG_SIZE

    def _debug_rb_lane_vals(self, cyclic_offset: int, source: RegFile) -> tuple[float, ...] | tuple[int, ...]:
        self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_offset)
        buf = source.get_r_cyclic_at(cyclic_offset, R_CYCLIC_SIZE)
        return self._wide_unpack_lane_tuple(buf)

    def _acc_agg_lane_fmt(self) -> str:
        """Struct format for r_acc / agg when wide-vector debug is on."""
        if self._wide_vector_active():
            return "<f" if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<i"
        dtype = self.state.get_cr_dtype()
        return "<i" if dtype == DType.INT8 else "<f"

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
        - AddSubSrcB → like LcrIdx for codes 0–31; codes ≥ 32 → unsigned IMM5 (low 5 bits)
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
        elif op_type == "AddSubSrcB":
            # 6-bit encoding: 0–31 same as LcrIdx; 32–63 → unsigned IMM5 (low 5 bits).
            if raw_value >= 32:
                return raw_value & 31
            if raw_value < LR_REG_COUNT:
                return source.get_lr(raw_value)
            return source.get_cr(raw_value - LR_REG_COUNT)
        elif op_type == "MultStageReg":
            if raw_value > 1:
                raise EmulatorError(
                    "Mult-stage operand must encode r0 (0) or r1 (1); "
                    f"got {raw_value}"
                )
            if self._wide_vector_active():
                # Mult handlers read wide lanes from _debug_mult_stage_vectors_snap
                # keyed by MultStageReg encoding index (0=r0, 1=r1).
                return raw_value
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
        if self._wide_vector_active():
            return

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
        """Execute str_acc_reg: Store accumulator to memory (debug only)."""
        warnings.warn(
            "[DEBUG ONLY] str_acc_reg is not a hardware instruction and is available "
            "for emulator debugging purposes only",
            stacklevel=2,
        )
        addr = offset + base
        acc_data = self.state.regfile.get_r_acc_bytes()
        self.state.xmem.write_address(addr, acc_data)

    def execute_ldr_mult_reg(self, *, dest: int, offset: int, base: int) -> None:
        """Execute ldr_mult_reg: Load data from memory into a mult stage register."""
        if dest not in (0, 1):
            raise EmulatorError(
                f"ldr_mult_reg: dest must be 0 (r0) or 1 (r1); got {dest}"
            )
        addr = offset + base
        if self._wide_vector_active():
            data = self.state.xmem.read_address(addr, R_CYCLIC_SIZE)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                self.state._debug_mult_stage_vectors[dest] = list(
                    struct.unpack_from("<128f", data, 0)
                )
            else:
                self.state._debug_mult_stage_vectors[dest] = list(
                    struct.unpack_from("<128i", data, 0)
                )
            return

        data = self.state.xmem.read_address(addr, R_REG_SIZE)
        reg_name, elem_idx = _MULT_STAGE_MAP[dest]
        self.state.regfile.set_register_bytes(reg_name, elem_idx, data)

    def execute_ldr_cyclic_mult_reg(self, *, offset: int, base: int, index: int) -> None:
        """Execute ldr_cyclic_mult_reg: Load with cyclic addressing into r_cyclic."""
        addr = offset + base
        assert index % R_REG_SIZE == 0, (
            f"LR index for cyclic load must be aligned to {R_REG_SIZE}: got {index}"
        )
        if self._wide_vector_active():
            assert index % R_CYCLIC_SIZE == 0, (
                f"Wide-vector debug: cyclic load index must be aligned to {R_CYCLIC_SIZE}, "
                f"got {index}"
            )
            data = self.state.xmem.read_address(addr, R_CYCLIC_SIZE)
            self.state.regfile.set_r_cyclic_at(index, data)
            return

        data = self.state.xmem.read_address(addr, R_REG_SIZE)
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

    def execute_lr_set(self, *, reg: int, value: int) -> None:
        """Execute set: Set a loop register to an immediate value."""
        self.state.regfile.set_lr(reg, self._sign_extend_16(value) & 0xFFFFFFFF)

    def execute_lr_add(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute add: uint32 ``dest = src_a + src_b`` (``src_b`` may be an immediate)."""
        self.state.regfile.set_lr(dest, (src_a + src_b) & 0xFFFFFFFF)

    def execute_lr_sub(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute sub: uint32 ``dest = src_a - src_b`` (``src_b`` may be an immediate)."""
        self.state.regfile.set_lr(dest, (src_a - src_b) & 0xFFFFFFFF)

    def execute_lr_incr_mod_pow2(self, *, dst: int, step: int, k: int) -> None:
        """incr_mod_pow2: dst <- (dst + step) mod 2^k.

        Old dst is taken from the cycle-start snapshot (read-before-write). Step is
        resolved from LcrIdx (snapshot), interpreted as uint32 like ``add``/``sub``.
        ``k`` is the raw encoded field (k_semantic − 1); semantic exponent is k + 1.
        """
        assert self.snapshot is not None
        if k > LR_MOD_POW2_K_ENCODED_MAX:
            raise EmulatorError(
                f"incr_mod_pow2: invalid k encoding {k} (max {LR_MOD_POW2_K_ENCODED_MAX})"
            )
        k_exp = k + LR_MOD_POW2_K_MIN
        cur = self.snapshot.get_lr(dst)
        step_u = step & 0xFFFFFFFF
        mask = (1 << k_exp) - 1
        self.state.regfile.set_lr(dst, ((cur + step_u) & 0xFFFFFFFF) & mask)

    def _dispatch_lr_slots(self, inst: dict[str, int]) -> None:
        """Dispatch all LR sub-slots with conflict detection.

        LR is special: the VLIW word contains multiple LR sub-instructions
        (lr_inst_0, lr_inst_1, …). Each is dispatched independently
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

            # Unfilled LR slots encode ``add lrX lrX 0`` (IMM5 0 → encoding 32): identity, no write.
            if inst_name == "add":
                sb = kwargs.get("src_b")
                if (
                    kwargs.get("dest") == kwargs.get("src_a")
                    and isinstance(sb, int)
                    and sb >= 32
                    and (sb & 31) == 0
                ):
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

    def execute_mult_ee(self, *, ra: bytearray | int, cyclic_offset: int,
                        mask_offset: int, mask_shift: int) -> None:
        """Execute mult_ee: Element-wise multiplication."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_offset)
            ra_vals = self._debug_ra_lane_vals(ra)
            rb_vals = self._debug_rb_lane_vals(cyclic_offset, self.state.regfile)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                for i in range(R_REG_SIZE):
                    struct.pack_into("<f", mult_res, i * 4, float(ra_vals[i]) * float(rb_vals[i]))
            else:
                for i in range(R_REG_SIZE):
                    struct.pack_into(
                        "<i", mult_res, i * 4, self._wide_imult32(int(ra_vals[i]), int(rb_vals[i]))
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        dtype = self.state.get_cr_dtype()
        rb = self.state.regfile.get_r_cyclic_at(cyclic_offset, R_REG_SIZE)

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ve(self, *, cyclic_offset: int,
                        mask_offset: int, mask_shift: int, fixed_idx: int) -> None:
        """Execute mult.ve: Fixed element from R0/R1 x RC elements with boundary padding.

        Uses fixed_idx to select the scalar multiplicand: indices 0-127 address
        R0[fixed_idx], indices 128-255 address R1[fixed_idx-128]. Multiplies that
        scalar against each byte of RC[cyclic_offset : cyclic_offset+128]. Non-cyclic:
        elements where cyclic_offset+i >= R_CYCLIC_SIZE are padded with the
        dtype-specific encoding of 1 instead of wrapping.
        """
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_offset)
            r0_vals = self._debug_ra_lane_vals(0)
            r1_vals = self._debug_ra_lane_vals(1)
            rb_vals = self._debug_rb_lane_vals(cyclic_offset, self.state.regfile)
            if fixed_idx < R_REG_SIZE:
                ra_fixed = r0_vals[fixed_idx % R_REG_SIZE]
            else:
                ra_fixed = r1_vals[(fixed_idx - R_REG_SIZE) % R_REG_SIZE]
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                one = 1.0
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = float(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into("<f", mult_res, i * 4, float(ra_fixed) * rb_lane)
            else:
                one = 1
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = int(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into(
                        "<i", mult_res, i * 4, self._wide_imult32(int(ra_fixed), rb_lane)
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        dtype = self.state.get_cr_dtype()
        r_buf = self.state.regfile.raw("r")  # 256 bytes: [0:128]=r0, [128:256]=r1
        rc_buf = self.state.regfile.raw("r_cyclic")
        one_byte = dtype_one_byte(dtype)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        ra_fixed = r_buf[fixed_idx % (2 * R_REG_SIZE)]

        for i in range(R_REG_SIZE):
            pos = cyclic_offset + i
            rb_byte = rc_buf[pos] if pos < R_CYCLIC_SIZE else one_byte
            result = ipu_mult(ra_fixed, rb_byte, dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ve_cr(self, *, cyclic_offset: int, mask_offset: int,
                           mask_shift: int, cr_idx: int) -> None:
        """Execute mult.ve.cr: CR scalar x RC elements with boundary padding.

        Multiplies the low byte of CR[cr_idx] against each byte of
        RC[cyclic_offset : cyclic_offset+128]. Unlike mult.ve, this is
        non-cyclic: elements where cyclic_offset+i >= R_CYCLIC_SIZE are
        padded with the dtype-specific encoding of 1 instead of wrapping.
        """
        dtype = self.state.get_cr_dtype()
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_offset)
            cr_scalar = self._wide_cr_scalar_byte_as_int32(cr_idx)
            rb_vals = self._debug_rb_lane_vals(cyclic_offset, self.state.regfile)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                scalar_f = float(cr_scalar)
                one = 1.0
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = float(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into("<f", mult_res, i * 4, scalar_f * rb_lane)
            else:
                one = 1
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = int(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into(
                        "<i", mult_res, i * 4, self._wide_imult32(cr_scalar, rb_lane)
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        scalar_byte = self.state.regfile.get_cr(cr_idx) & 0xFF
        rc_buf = self.state.regfile.raw("r_cyclic")
        one_byte = dtype_one_byte(dtype)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            pos = cyclic_offset + i
            rb_byte = rc_buf[pos] if pos < R_CYCLIC_SIZE else one_byte
            result = ipu_mult(scalar_byte, rb_byte, dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ve_aaq(self, *, cyclic_offset: int, mask_offset: int,
                            mask_shift: int, aaq_rf_idx: int) -> None:
        """Execute mult.ve.aaq: AAQ scalar x RC elements with boundary padding.

        Multiplies the low byte of AAQ[aaq_rf_idx] against each byte of
        RC[cyclic_offset : cyclic_offset+128]. Non-cyclic: elements where
        cyclic_offset+i >= R_CYCLIC_SIZE are padded with the dtype-specific
        encoding of 1 instead of wrapping.
        """
        dtype = self.state.get_cr_dtype()
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_offset)
            aaq_lane = self._wide_aaq_scalar(aaq_rf_idx)
            rb_vals = self._debug_rb_lane_vals(cyclic_offset, self.state.regfile)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                one = 1.0
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = float(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into("<f", mult_res, i * 4, float(aaq_lane) * rb_lane)
            else:
                one = 1
                for i in range(R_REG_SIZE):
                    pos = cyclic_offset + i * 4
                    rb_lane = int(rb_vals[i]) if pos + 4 <= R_CYCLIC_SIZE else one
                    struct.pack_into(
                        "<i", mult_res, i * 4, self._wide_imult32(int(aaq_lane), rb_lane)
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        scalar_byte = self.state.regfile.get_aaq(aaq_rf_idx) & 0xFF
        rc_buf = self.state.regfile.raw("r_cyclic")
        one_byte = dtype_one_byte(dtype)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(R_REG_SIZE):
            pos = cyclic_offset + i
            rb_byte = rc_buf[pos] if pos < R_CYCLIC_SIZE else one_byte
            result = ipu_mult(scalar_byte, rb_byte, dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

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
        fmt = self._acc_agg_lane_fmt()

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            if self._wide_vector_active():
                result = self._wide_add_lane(acc_val, mult_val)
            else:
                result = ipu_add(acc_val, mult_val, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_first(self) -> None:
        """Execute acc.first: Set r_acc to multiply result (no previous sum)."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        fmt = self._acc_agg_lane_fmt()

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            struct.pack_into(fmt, acc_buf, i * 4, mult_val)

    def execute_acc_add_aaq(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.add_aaq: Accumulate mult_res, then add aaq[aaq_rf_idx] to each of the 128 accumulator words."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self.snapshot.raw("r_acc")
        fmt = self._acc_agg_lane_fmt()
        if self._wide_vector_active():
            aaq_lane = self._wide_aaq_scalar(aaq_rf_idx)
        else:
            aaq_lane = self.state.regfile.get_aaq(aaq_rf_idx)

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            if self._wide_vector_active():
                result = self._wide_add_lane(self._wide_add_lane(acc_val, mult_val), aaq_lane)
            else:
                result = ipu_add(ipu_add(acc_val, mult_val, dtype), aaq_lane, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_add_aaq_first(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.add_aaq.first: Set r_acc to mult_res + aaq[aaq_rf_idx] (no previous sum)."""
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        fmt = self._acc_agg_lane_fmt()
        if self._wide_vector_active():
            aaq_lane = self._wide_aaq_scalar(aaq_rf_idx)
        else:
            aaq_lane = self.state.regfile.get_aaq(aaq_rf_idx)

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            if self._wide_vector_active():
                result = self._wide_add_lane(mult_val, aaq_lane)
            else:
                result = ipu_add(mult_val, aaq_lane, dtype)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_max(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.max: r_acc[i] = max(r_acc[i], mult_res[i], aaq_reg[aaq_rf_idx]).

        All register values are interpreted as signed (int32 for INT8 dtype, float32 for FP8).
        """
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self.snapshot.raw("r_acc")
        aaq_raw = self.state.regfile.get_aaq(aaq_rf_idx) & 0xFFFFFFFF
        fmt = self._acc_agg_lane_fmt()
        aaq_val = struct.unpack(fmt, struct.pack("<I", aaq_raw))[0]

        for i in range(R_REG_SIZE):
            acc_val = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = max(acc_val, mult_val, aaq_val)
            struct.pack_into(fmt, acc_buf, i * 4, result)

    def execute_acc_max_first(self, *, aaq_rf_idx: int) -> None:
        """Execute acc.max.first: r_acc[i] = max(mult_res[i], aaq_reg[aaq_rf_idx]). Previous r_acc ignored.

        All register values are interpreted as signed (int32 for INT8 dtype, float32 for FP8).
        """
        dtype = self.state.get_cr_dtype()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        aaq_raw = self.state.regfile.get_aaq(aaq_rf_idx) & 0xFFFFFFFF
        fmt = self._acc_agg_lane_fmt()
        aaq_val = struct.unpack(fmt, struct.pack("<I", aaq_raw))[0]

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            result = max(mult_val, aaq_val)
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
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")

        for i, idx in enumerate(out_indices):
            if idx >= 0:
                val = struct.unpack_from(fmt, mult_res, idx * 4)[0]
            else:
                val = 0.0 if fmt == "<f" else 0
            struct.pack_into(fmt, acc_buf, (base + i) * 4, val)

    def execute_aaq_nop(self) -> None:
        """Execute aaq_nop: No operation for AAQ slot."""
        pass

    def execute_agg(
        self,
        *,
        agg_mode: int,
        post_fn: int,
        cr_idx: int,
        aaq_rf_idx: int,
    ) -> None:
        """Execute acc.agg: Collapse 128 r_acc words to one value (SUM or MAX), apply post function, store to AAQ.

        For MAX, the current value of the target AAQ register is included in the max (no update if already max).
        """
        dtype = self.state.get_cr_dtype()
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.state.regfile.raw("r_acc")
        n_words = R_ACC_SIZE // 4  # 128

        # Collect all 128 words as scalar values
        values = []
        for i in range(n_words):
            val = struct.unpack_from(fmt, acc_buf, i * 4)[0]
            values.append(val)

        if get_agg_mode(agg_mode) == AGG_MODE_SUM:
            raw_result = sum(values)
        else:
            # MAX: include current AAQ value in the comparison
            aaq_raw = self.state.regfile.get_aaq(aaq_rf_idx) & 0xFFFFFFFF
            current_aaq = struct.unpack(fmt, struct.pack("<I", aaq_raw))[0]
            raw_result = max(values + [current_aaq])

        # Apply post function
        fn = get_post_fn(post_fn)
        if fn == POST_FN_VALUE:
            result_val = raw_result
        elif fn == POST_FN_VALUE_CR:
            cr_val = self.state.regfile.get_cr(cr_idx) & 0xFFFFFFFF
            cr_scalar = struct.unpack(fmt, struct.pack("<I", cr_val))[0]
            if fmt == "<i":
                result_val = int(raw_result) * int(cr_scalar)
                p = result_val & 0xFFFFFFFF
                result_val = p - 0x100000000 if p >= 0x80000000 else p
            else:
                result_val = raw_result * cr_scalar
        elif fn == POST_FN_INV:
            if dtype == DType.INT8 and not self._wide_vector_active():
                # Integer path: avoid div by zero; use float then store as float bits
                f = float(raw_result)
                result_val = 1.0 / f if f != 0 else 0.0
                # Store as float32 bit pattern
                out_bits = struct.unpack("<I", struct.pack("<f", result_val))[0]
                self.state.regfile.set_aaq(aaq_rf_idx, out_bits)
                return
            else:
                if fmt == "<i":
                    ri = int(raw_result)
                    result_val = 0 if ri == 0 else int(round(1.0 / float(ri)))
                else:
                    result_val = 1.0 / raw_result if raw_result != 0 else 0.0
        elif fn == POST_FN_INV_SQRT:
            if dtype == DType.INT8 and not self._wide_vector_active():
                f = float(raw_result)
                result_val = 1.0 / (f ** 0.5) if f > 0 else 0.0
                out_bits = struct.unpack("<I", struct.pack("<f", result_val))[0]
                self.state.regfile.set_aaq(aaq_rf_idx, out_bits)
                return
            else:
                if fmt == "<i":
                    ri = int(raw_result)
                    result_val = (
                        int(round(1.0 / (float(ri) ** 0.5))) if ri > 0 else 0
                    )
                else:
                    result_val = 1.0 / (raw_result ** 0.5) if raw_result > 0 else 0.0
        else:
            result_val = raw_result

        self.state.regfile.set_aaq(aaq_rf_idx, self._wide_pack_aaq_bits(fmt, result_val))

    def execute_agg_first(
        self,
        *,
        agg_mode: int,
        post_fn: int,
        cr_idx: int,
        aaq_rf_idx: int,
    ) -> None:
        """Execute agg.first: like agg but for MAX mode ignores previous AAQ value."""
        dtype = self.state.get_cr_dtype()
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.state.regfile.raw("r_acc")
        n_words = R_ACC_SIZE // 4  # 128

        values = []
        for i in range(n_words):
            val = struct.unpack_from(fmt, acc_buf, i * 4)[0]
            values.append(val)

        if get_agg_mode(agg_mode) == AGG_MODE_SUM:
            raw_result = sum(values)
        else:
            # MAX: do NOT include previous AAQ value — this is the .first behaviour
            raw_result = max(values)

        fn = get_post_fn(post_fn)
        if fn == POST_FN_VALUE:
            result_val = raw_result
        elif fn == POST_FN_VALUE_CR:
            cr_val = self.state.regfile.get_cr(cr_idx) & 0xFFFFFFFF
            cr_scalar = struct.unpack(fmt, struct.pack("<I", cr_val))[0]
            if fmt == "<i":
                result_val = int(raw_result) * int(cr_scalar)
                p = result_val & 0xFFFFFFFF
                result_val = p - 0x100000000 if p >= 0x80000000 else p
            else:
                result_val = raw_result * cr_scalar
        elif fn == POST_FN_INV:
            if dtype == DType.INT8 and not self._wide_vector_active():
                f = float(raw_result)
                result_val = 1.0 / f if f != 0 else 0.0
                out_bits = struct.unpack("<I", struct.pack("<f", result_val))[0]
                self.state.regfile.set_aaq(aaq_rf_idx, out_bits)
                return
            else:
                if fmt == "<i":
                    ri = int(raw_result)
                    result_val = 0 if ri == 0 else int(round(1.0 / float(ri)))
                else:
                    result_val = 1.0 / raw_result if raw_result != 0 else 0.0
        elif fn == POST_FN_INV_SQRT:
            if dtype == DType.INT8 and not self._wide_vector_active():
                f = float(raw_result)
                result_val = 1.0 / (f ** 0.5) if f > 0 else 0.0
                out_bits = struct.unpack("<I", struct.pack("<f", result_val))[0]
                self.state.regfile.set_aaq(aaq_rf_idx, out_bits)
                return
            else:
                if fmt == "<i":
                    ri = int(raw_result)
                    result_val = (
                        int(round(1.0 / (float(ri) ** 0.5))) if ri > 0 else 0
                    )
                else:
                    result_val = 1.0 / (raw_result ** 0.5) if raw_result > 0 else 0.0
        else:
            result_val = raw_result

        self.state.regfile.set_aaq(aaq_rf_idx, self._wide_pack_aaq_bits(fmt, result_val))

    def execute_aaq(self) -> None:
        """Execute aaq: Quantize r_acc (128 × INT32) → aaq_result (128 × INT8).

        Requires INT8 mode. Each 32-bit word is truncated (top 8 bits taken via
        arithmetic right-shift by 24) then clamped to [-128, 127] and stored as
        a signed byte in the aaq_result register.

        In wide-vector debug mode, ``aaq`` is normally a no-op (results stay in
        ``r_acc`` as 32-bit lanes; use ``str_acc_reg`` to dump them). Set
        ``state.wide_vector_quantize_output`` to re-run INT8-style quantization
        from wide lanes for comparison with the real path.
        """
        dtype = self.state.get_cr_dtype()
        if dtype != DType.INT8:
            raise EmulatorError("AAQ instruction requires INT8 mode")

        if self._wide_vector_active():
            if not self.state.wide_vector_quantize_output:
                return
            acc_buf = self.state.regfile.raw("r_acc")
            result = bytearray(128)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                for i in range(128):
                    val = struct.unpack_from("<f", acc_buf, i * 4)[0]
                    clamped = max(-128, min(127, int(round(val))))
                    result[i] = clamped & 0xFF
            else:
                for i in range(128):
                    val = struct.unpack_from("<i", acc_buf, i * 4)[0]
                    truncated = val >> 24
                    clamped = max(-128, min(127, truncated))
                    result[i] = clamped & 0xFF
            self.state.regfile.set_aaq_result(result)
            return

        acc_buf = self.state.regfile.raw("r_acc")
        result = bytearray(128)
        for i in range(128):
            val = struct.unpack_from("<i", acc_buf, i * 4)[0]
            truncated = val >> 24  # arithmetic right-shift: keeps top 8 bits, range [-128, 127]
            clamped = max(-128, min(127, truncated))
            result[i] = clamped & 0xFF
        self.state.regfile.set_aaq_result(result)

    def execute_xmem_store_aaq_result(self, *, offset: int, base: int) -> None:
        """Execute xmem.store_aaq_result: Write 128-byte aaq_result to xmem."""
        addr = offset + base
        data = self.state.regfile.get_aaq_result()
        self.state.xmem.write_address(addr, data)

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
        if self._wide_vector_active():
            self.state._debug_mult_stage_vectors_snap = {
                k: list(v) for k, v in self.state._debug_mult_stage_vectors.items()
            }

        # Break runs first — may halt before side effects
        result = self.dispatch_instruction("break", inst)
        if result == BreakResult.BREAK:
            return BreakResult.BREAK

        # Execute all other slots using the snapshot
        self._dispatch_lr_slots(inst)  # LR has multiple sub-slots
        self.dispatch_instruction("xmem", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("aaq", inst)
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
        if self._wide_vector_active():
            self.state._debug_mult_stage_vectors_snap = {
                k: list(v) for k, v in self.state._debug_mult_stage_vectors.items()
            }

        # Execute all slots except break
        self._dispatch_lr_slots(inst)
        self.dispatch_instruction("xmem", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("aaq", inst)
        self.dispatch_instruction("cond", inst)
