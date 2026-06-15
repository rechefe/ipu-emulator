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
from ipu_emu.ipu_math import ipu_mult, ipu_add, ipu_sub, dtype_one_byte, DType
from ipu_emu.ipu_config import REGISTER_WORD_VALUE_MASK, LR_CR_SCALAR_BITS, Partition
from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    SLOT_BINARY_LAYOUT,
    SLOT_UNIONS,
    SLOT_COUNT,
    get_instruction_by_opcode,
    create_emulator_constants,
)
from ipu_common.acc_stride_enums import (
    get_elements_per_row,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_common.incr_mod_pow2_k import LR_MOD_POW2_K_ENCODED_MAX, LR_MOD_POW2_K_MIN
from ipu_common.registers import get_register_sizes, get_mult_stage_map
from ipu_common.activations import apply_activation

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
    "ElementsInRow": "elements_in_row_field",
    "HorizontalStride": "horizontal_stride_field",
    "VerticalStride": "vertical_stride_field",
    "LrModPow2KImmediate": "lr_mod_pow2_k_immediate",
    "MultMaskOffsetImmediate": "mult_mask_offset_immediate",
    "ActivationFn": "activation_fn_field",
    "BreakImmediate": "break_immediate_type",
    "Label": "label_token",
    "FullXmemRow": "full_xmem_row_field",
}

# Field prefix for each slot type (matches compound_inst naming)
_SLOT_FIELD_PREFIX = {
    "load": "load_inst",
    "store": "store_inst",
    "acc_store": "acc_store_inst",
    "mult": "mult_inst",
    "acc": "acc_inst",
    "aaq": "aaq_inst",
    "cond": "cond_inst",
    "break": "break_inst",
    # LR is special: "lr_inst_0", "lr_inst_1", ... (count from SLOT_COUNT["lr"])
}


def _build_field_map_for_instruction_union(
    prefix: str, slot_union: object, inst_name: str,
) -> dict[str, str]:
    """Build operand_name → inst dict field_key mapping using union bindings."""
    field_map: dict[str, str] = {}
    for field_idx, operand_name in slot_union.opcode_bindings.get(inst_name, []):
        canonical_type = slot_union.fields[field_idx].canonical_type
        token_idx = field_idx + 1  # +1: token_0 is the opcode
        suffix = _TYPE_FIELD_SUFFIX[canonical_type]
        field_map[operand_name] = f"{prefix}_token_{token_idx}_{suffix}"
    return field_map


def _build_all_instruction_field_maps() -> dict[tuple, dict[str, str]]:
    """Precompute operand → field_key mappings for ALL instructions.

    Returns a dict keyed by:
      (slot_type, inst_name)       — for non-LR slots
      ("lr", inst_name, slot_idx)  — for LR sub-slots (0 .. SLOT_COUNT["lr"]-1)
    """
    result: dict[tuple, dict[str, str]] = {}

    for slot_type, prefix in _SLOT_FIELD_PREFIX.items():
        slot_union = SLOT_UNIONS[slot_type]
        for inst_name in INSTRUCTION_SPEC[slot_type]:
            result[(slot_type, inst_name)] = _build_field_map_for_instruction_union(
                prefix, slot_union, inst_name
            )

    # LR has multiple sub-slots with different field prefixes
    lr_slot_union = SLOT_UNIONS["lr"]
    for inst_name in INSTRUCTION_SPEC["lr"]:
        for slot_idx in range(SLOT_COUNT["lr"]):
            prefix = f"lr_inst_{slot_idx}"
            result[("lr", inst_name, slot_idx)] = _build_field_map_for_instruction_union(
                prefix, lr_slot_union, inst_name
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

    def _wide_sub_lane(self, a: float | int, b: float | int) -> float | int:
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            return float(a) - float(b)
        return ipu_sub(int(a), int(b), DType.INT8)

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
        dtype = self.state.dtype
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

    @staticmethod
    def _build_partition_vector(num_partitions: int) -> int:
        """Build the left-shift partition vector (0 at the START of each group).

        Used for positive mask_shift indices (+1, +2, +3).
        num_partitions must be in VALID_PARTITION_VALUES.
        num_partitions=0: all-ones — no boundaries, shifts are unconstrained.
        num_partitions=P: P groups of R_REG_SIZE/P lanes; bit 0 of each group is 0.
        """
        assert isinstance(num_partitions, Partition), (
            f"partition must be a Partition enum value, got {num_partitions!r}"
        )
        if num_partitions == 0:
            return (1 << R_REG_SIZE) - 1
        step = R_REG_SIZE // num_partitions
        result = 0
        for i in range(R_REG_SIZE):
            if i % step != 0:
                result |= (1 << i)
        return result

    @staticmethod
    def _build_inverse_partition_vector(num_partitions: int) -> int:
        """Build the right-shift partition vector (0 at the END of each group).

        Used for negative mask_shift indices (−1, −2, −3).
        num_partitions must be in VALID_PARTITION_VALUES.
        num_partitions=0: all-ones — no boundaries, shifts are unconstrained.
        num_partitions=P: P groups of R_REG_SIZE/P lanes; last bit of each group is 0.
        """
        assert isinstance(num_partitions, Partition), (
            f"partition must be a Partition enum value, got {num_partitions!r}"
        )
        if num_partitions == 0:
            return (1 << R_REG_SIZE) - 1
        step = R_REG_SIZE // num_partitions
        result = 0
        for i in range(R_REG_SIZE):
            if i % step != step - 1:
                result |= (1 << i)
        return result

    def _mult_mask_and_shift(self, mask_idx: int, shift: int) -> None:
        """Apply sequential shift-and-AND mask generation, then gate mult_res.

        ``shift`` is interpreted as ``mask_shift_idx`` ∈ [−3, +3] (clamped).
        The slot mask M is taken from R_MASK slot ``mask_idx``, then shifted
        sequentially (one bit per step):
          idx 0  → M (unmodified)
          idx +k → shift left  k times, ANDing with partition_vector after each step
          idx −k → shift right k times, ANDing with inverse_partition_vector after each step

        Two partition vectors, both derived from CR15.partition:
          partition_vector         — 0 at the START of each group (used for left shifts)
          inverse_partition_vector — 0 at the END   of each group (used for right shifts)

        Lanes in mult_res where the resulting mask bit is 0 are zeroed.
        """
        if self._wide_vector_active():
            return

        # LR registers are LR_CR_SCALAR_BITS wide; sign-extend before clamping
        if shift >= (1 << (LR_CR_SCALAR_BITS - 1)):
            shift = shift - (1 << LR_CR_SCALAR_BITS)
        shift = max(-3, min(3, shift))

        # Extract 128-bit base mask from the selected R_MASK slot
        mask_bytes = self.state.regfile.get_r_mask()
        mask_slot = mask_idx % (R_REG_SIZE // 16)  # 8 slots of 128 bits each
        offset = mask_slot * 16
        _128_BIT_MASK = (1 << R_REG_SIZE) - 1
        base_mask = int.from_bytes(mask_bytes[offset:offset + 16], byteorder="little") & _128_BIT_MASK

        num_partitions = self.state.get_config_partition()

        # Generate the shifted mask via sequential shift-and-AND
        mask_int = base_mask
        if shift < 0:
            pv = self._build_inverse_partition_vector(num_partitions)
            for _ in range(-shift):
                mask_int = (mask_int >> 1) & pv
        elif shift > 0:
            pv = self._build_partition_vector(num_partitions)
            for _ in range(shift):
                mask_int = (mask_int << 1) & pv & _128_BIT_MASK

        # Zero out mult_res where mask bit is clear (lane deactivated)
        mult_res = self.state.regfile.raw("mult_res")
        for i in range(R_REG_SIZE):
            if not ((mask_int >> i) & 1):
                struct.pack_into("<I", mult_res, i * 4, 0)

    # -----------------------------------------------------------------------
    # Memory slot instruction handlers (load / store / acc_store)
    # -----------------------------------------------------------------------

    def execute_load_nop(self) -> None:
        """Execute LOAD_NOP: No operation."""
        pass

    def execute_store_nop(self) -> None:
        """Execute STORE_NOP: No operation."""
        pass

    def execute_acc_store_nop(self) -> None:
        """Execute ACC_STORE_NOP: No operation."""
        pass

    def execute_str_acc_reg(self, *, offset: int, base: int) -> None:
        """Execute STR_ACC_REG: Store accumulator to memory (debug only)."""
        warnings.warn(
            "[DEBUG ONLY] STR_ACC_REG is not a hardware instruction and is available "
            "for emulator debugging purposes only",
            stacklevel=2,
        )
        addr = offset + base
        acc_data = self.state.regfile.get_r_acc_bytes()
        self.state.xmem.write_address(addr, acc_data)

    def execute_ldr_mult_reg(self, *, dest: int, offset: int, base: int) -> None:
        """Execute LDR_MULT_REG: Load data from memory into a mult stage register."""
        if dest not in (0, 1):
            raise EmulatorError(
                f"LDR_MULT_REG: dest must be 0 (r0) or 1 (r1); got {dest}"
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
        """Execute LDR_CYCLIC_MULT_REG: Load with cyclic addressing into r_cyclic."""
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
        """Execute LDR_MULT_MASK_REG: Load mask data from memory."""
        addr = offset + base
        data = self.state.xmem.read_address(addr, R_REG_SIZE)
        self.state.regfile.set_r_mask(data)

    # -----------------------------------------------------------------------
    # LR Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_lr_set(self, *, reg: int, src: int) -> None:
        """Execute SET: Copy a 32-bit value from a configuration register into an LR."""
        self.state.regfile.set_lr(reg, src & 0xFFFFFFFF)

    def execute_lr_add(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute ADD: uint32 ``dest = src_a + src_b`` (``src_b`` may be an immediate)."""
        self.state.regfile.set_lr(dest, (src_a + src_b) & 0xFFFFFFFF)

    def execute_lr_sub(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute SUB: uint32 ``dest = src_a - src_b`` (``src_b`` may be an immediate)."""
        self.state.regfile.set_lr(dest, (src_a - src_b) & 0xFFFFFFFF)

    def execute_lr_incr_mod_pow2(self, *, dest: int, step: int, k: int) -> None:
        """INCR_MOD_POW2: dest <- (dest + step) mod 2^k.

        Old dest is taken from the cycle-start snapshot (read-before-write). Step is
        resolved from LcrIdx (snapshot), interpreted as uint32 like ``add``/``sub``.
        ``k`` is the raw encoded field (k_semantic − 1); semantic exponent is k + 1.
        """
        assert self.snapshot is not None
        if k > LR_MOD_POW2_K_ENCODED_MAX:
            raise EmulatorError(
                f"INCR_MOD_POW2: invalid k encoding {k} (max {LR_MOD_POW2_K_ENCODED_MAX})"
            )
        k_exp = k + LR_MOD_POW2_K_MIN
        cur = self.snapshot.get_lr(dest)
        step_u = step & 0xFFFFFFFF
        mask = (1 << k_exp) - 1
        self.state.regfile.set_lr(dest, ((cur + step_u) & 0xFFFFFFFF) & mask)

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

            # Unfilled LR slots encode ``ADD lrX lrX 0`` (IMM5 0 → encoding 32): identity, no write.
            if inst_name == "ADD":
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
        real_targets = [t for t in lr_targets if t is not None]
        if len(real_targets) != len(set(real_targets)):
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
        """Execute MULT_NOP: No operation."""
        pass

    def execute_mult_ee(self, *, ra: bytearray | int, cyclic_offset: int,
                        mask_offset: int, mask_shift: int) -> None:
        """Execute MULT.EE: Element-wise multiplication."""
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

        dtype = self.state.dtype
        rb = self.state.regfile.get_r_cyclic_at(cyclic_offset, R_REG_SIZE)

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], rb[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ee_rr(self, *, ra: bytearray | int,
                           mask_offset: int, mask_shift: int) -> None:
        """Execute MULT.EE.RR: multi-element multiply of a mult-stage register by itself.

        ``ra`` selects the MEE mode: R0 → r0-by-r0, R1 → r1-by-r1. Each lane is
        multiplied by itself (element-wise square), then masked and shifted.
        """
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            ra_vals = self._debug_ra_lane_vals(ra)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                for i in range(R_REG_SIZE):
                    struct.pack_into(
                        "<f", mult_res, i * 4, float(ra_vals[i]) * float(ra_vals[i])
                    )
            else:
                for i in range(R_REG_SIZE):
                    struct.pack_into(
                        "<i", mult_res, i * 4,
                        self._wide_imult32(int(ra_vals[i]), int(ra_vals[i])),
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        dtype = self.state.dtype

        for i in range(R_REG_SIZE):
            result = ipu_mult(ra[i], ra[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def _execute_mult_ve_variant(
        self,
        *,
        pad_128_ones: bool,
        cyclic_offset: int,
        mask_offset: int,
        mask_shift: int,
        fixed_idx: int,
    ) -> None:
        """Shared mult.ve.cyclic / mult.ve.padded implementation."""
        raw = cyclic_offset & 0xFFFFFFFF
        co_cyclic = raw % R_CYCLIC_SIZE

        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            co_wb = raw if pad_128_ones else co_cyclic
            self._wide_assert_lane_aligned_byte_offset("cyclic_offset", co_wb)
            r0_vals = self._debug_ra_lane_vals(0)
            r1_vals = self._debug_ra_lane_vals(1)
            rb_vals = self._debug_rb_lane_vals(co_wb, self.state.regfile)
            if fixed_idx < R_REG_SIZE:
                ra_fixed = r0_vals[fixed_idx % R_REG_SIZE]
            else:
                ra_fixed = r1_vals[(fixed_idx - R_REG_SIZE) % R_REG_SIZE]
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                one = 1.0
                for i in range(R_REG_SIZE):
                    pos = co_wb + i * 4
                    if pad_128_ones and pos + 4 > R_CYCLIC_SIZE:
                        rb_lane = one
                    else:
                        rb_lane = float(rb_vals[i])
                    struct.pack_into("<f", mult_res, i * 4, float(ra_fixed) * rb_lane)
            else:
                one = 1
                for i in range(R_REG_SIZE):
                    pos = co_wb + i * 4
                    if pad_128_ones and pos + 4 > R_CYCLIC_SIZE:
                        rb_lane = one
                    else:
                        rb_lane = int(rb_vals[i])
                    struct.pack_into(
                        "<i", mult_res, i * 4, self._wide_imult32(int(ra_fixed), rb_lane)
                    )
            self._mult_mask_and_shift(mask_offset, mask_shift)
            return

        dtype = self.state.dtype
        r_buf = self.state.regfile.raw("r")  # 256 bytes: [0:128]=r0, [128:256]=r1
        rc_buf = self.state.regfile.raw("r_cyclic")
        one_byte = dtype_one_byte(dtype)
        fmt = "<i" if dtype == DType.INT8 else "<f"

        ra_fixed = r_buf[fixed_idx % (2 * R_REG_SIZE)]

        if pad_128_ones:
            for i in range(R_REG_SIZE):
                pos = raw + i
                rb_byte = rc_buf[pos] if pos < R_CYCLIC_SIZE else one_byte
                result = ipu_mult(ra_fixed, rb_byte, dtype)
                struct.pack_into(fmt, mult_res, i * 4, result)
        else:
            base = co_cyclic
            for i in range(R_REG_SIZE):
                pos = base + i
                rb_byte = rc_buf[pos % R_CYCLIC_SIZE]
                result = ipu_mult(ra_fixed, rb_byte, dtype)
                struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift)

    def execute_mult_ve_cyclic(self, *, cyclic_offset: int,
                               mask_offset: int, mask_shift: int, fixed_idx: int) -> None:
        """Execute MULT.VE.CYCLIC: fixed r0/r1 element × r_cyclic row with cyclic addressing."""
        self._execute_mult_ve_variant(
            pad_128_ones=False,
            cyclic_offset=cyclic_offset,
            mask_offset=mask_offset,
            mask_shift=mask_shift,
            fixed_idx=fixed_idx,
        )

    def execute_mult_ve_padded(self, *, cyclic_offset: int,
                               mask_offset: int, mask_shift: int, fixed_idx: int) -> None:
        """Execute MULT.VE.PADDED: fixed r0/r1 element × r_cyclic row with boundary padding."""
        self._execute_mult_ve_variant(
            pad_128_ones=True,
            cyclic_offset=cyclic_offset,
            mask_offset=mask_offset,
            mask_shift=mask_shift,
            fixed_idx=fixed_idx,
        )

    def execute_mult_ve_cr(self, *, cyclic_offset: int, mask_offset: int,
                           mask_shift: int, cr_idx: int) -> None:
        """Execute MULT.VE.CR: CR scalar × r_cyclic elements with boundary padding.

        Multiplies the low byte of CR[cr_idx] against each byte of
        RC[cyclic_offset : cyclic_offset+128]. Like mult.ve.padded, this is
        non-cyclic: elements where cyclic_offset+i >= R_CYCLIC_SIZE are
        padded with the dtype-specific encoding of 1 instead of wrapping.
        """
        dtype = self.state.dtype
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

    # -----------------------------------------------------------------------
    # ACC Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_acc_nop(self) -> None:
        """Execute ACC_NOP: No operation."""
        pass

    def execute_reset_acc(self) -> None:
        """Execute RESET_ACC: Reset accumulator to zero."""
        self.state.regfile.set_r_acc_bytes(bytearray(R_ACC_SIZE))

    def execute_acc(self) -> None:
        """Execute ACC: Accumulate mult_res into accumulator."""
        dtype = self.state.dtype
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
        """Execute ACC.FIRST: Set r_acc to multiply result (no previous sum)."""
        dtype = self.state.dtype
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        fmt = self._acc_agg_lane_fmt()

        for i in range(R_REG_SIZE):
            mult_val = struct.unpack_from(fmt, mult_res, i * 4)[0]
            struct.pack_into(fmt, acc_buf, i * 4, mult_val)

    def execute_acc_stride(
        self,
        *,
        elements_in_row: int,
        horizontal_stride: int,
        vertical_stride: int,
        offset: int,
    ) -> None:
        """Execute ACC.STRIDE: Decimate mult_res by horizontal/vertical stride and write into r_acc.

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
        dtype = self.state.dtype
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
        """Execute AAQ_NOP: No operation for AAQ slot."""
        pass

    def _agg_active_lane_count(self, valid_elements: int) -> int:
        """Number of r_acc words included in aggregation (clamped to 128)."""
        n_words = R_ACC_SIZE // 4
        v = int(valid_elements) & 0xFFFFFFFF
        return min(v, n_words)

    @staticmethod
    def _to_int32(val: int) -> int:
        v = int(val) & 0xFFFFFFFF
        return v - 0x100000000 if v >= 0x80000000 else v

    def _agg_sum_lanes(self, fmt: str, snap_acc: bytearray, active: int) -> float | int:
        total: float | int = 0.0 if fmt == "<f" else 0
        for i in range(active):
            total += struct.unpack_from(fmt, snap_acc, i * 4)[0]
        return total

    def _agg_max_lanes(
        self, fmt: str, snap_acc: bytearray, active: int, seed: float | int
    ) -> float | int:
        best = seed
        for i in range(active):
            v = struct.unpack_from(fmt, snap_acc, i * 4)[0]
            if v > best:
                best = v
        return best

    def execute_agg_sum_first(self, *, dest_slot: int, full_xmem_row: int) -> None:
        """Execute AGG.SUM.FIRST: sum active R_ACC lanes, write to R_ACC[dest] (clean init)."""
        valid_elements = 128 if full_xmem_row else self.state.get_config_valid_elements()
        fmt = self._acc_agg_lane_fmt()
        snap_acc = self.snapshot.raw("r_acc")
        active = self._agg_active_lane_count(valid_elements)
        result = self._agg_sum_lanes(fmt, snap_acc, active)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        if fmt == "<i":
            result = self._to_int32(result)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_sum(self, *, dest_slot: int, full_xmem_row: int) -> None:
        """Execute AGG.SUM: sum active R_ACC lanes and add to R_ACC[dest] (running accumulation)."""
        valid_elements = 128 if full_xmem_row else self.state.get_config_valid_elements()
        fmt = self._acc_agg_lane_fmt()
        snap_acc = self.snapshot.raw("r_acc")
        active = self._agg_active_lane_count(valid_elements)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        snap_dest = struct.unpack_from(fmt, snap_acc, dest * 4)[0]
        partial = self._agg_sum_lanes(fmt, snap_acc, active)
        if fmt == "<f":
            result: float | int = float(partial) + float(snap_dest)
        else:
            result = ipu_add(self._to_int32(partial), int(snap_dest), DType.INT8)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_max_first(self, *, dest_slot: int, full_xmem_row: int) -> None:
        """Execute AGG.MAX.FIRST: max of active R_ACC lanes, write to R_ACC[dest] (no seed).

        When no lanes are active (valid_elements=0) the identity seed
        (INT32_MIN / -inf) is written, so the destination is always defined.
        """
        valid_elements = 128 if full_xmem_row else self.state.get_config_valid_elements()
        fmt = self._acc_agg_lane_fmt()
        snap_acc = self.snapshot.raw("r_acc")
        active = self._agg_active_lane_count(valid_elements)
        seed: float | int = -2147483648 if fmt == "<i" else float("-inf")
        result = self._agg_max_lanes(fmt, snap_acc, active, seed)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_max(self, *, dest_slot: int, full_xmem_row: int) -> None:
        """Execute AGG.MAX: max of active R_ACC lanes seeded with R_ACC[dest] (running max)."""
        valid_elements = 128 if full_xmem_row else self.state.get_config_valid_elements()
        fmt = self._acc_agg_lane_fmt()
        snap_acc = self.snapshot.raw("r_acc")
        active = self._agg_active_lane_count(valid_elements)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        snap_dest = struct.unpack_from(fmt, snap_acc, dest * 4)[0]
        result = self._agg_max_lanes(fmt, snap_acc, active, snap_dest)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_aaq(self, *, full_xmem_row: int) -> None:
        """Execute AAQ: Quantize wide lanes in ``POST_AAQ_REG`` (128 × 32-bit) → leading bytes.

        Source is the **512-byte** ``post_aaq_reg`` buffer (same lane layout as
        ``r_acc``), typically filled by ``ACTIVATE``. Each active 32-bit lane is
        clamped to the INT8 range ``[-128, 127]`` and stored in the leading bytes
        of ``post_aaq_reg``; the remainder is zeroed.

        Quantization is an **interim direct clamp** of the post-``ACTIVATE``
        accumulator lane: ``out[i] = clamp(lane[i], -128, 127)``.  The previous
        ``lane >> 24`` truncation was the shift half of a fixed-point requantize
        whose per-lane multiply half is not implemented, so on a real INT8
        convolution accumulator (magnitudes well below ``2**24``) it produced an
        all-zero result.

        This clamp is a placeholder, not the final design.  The intended future
        quantization is a full per-128-element (per-lane) requantize — applied
        after ``ACTIVATE`` and before write-back to memory — that scales each
        lane so the result is guaranteed to land in INT8 range; that stage
        subsumes (and will replace) this clamp.  Until it lands, clamping keeps
        the ``ACTIVATE`` -> ``AAQ`` path producing usable INT8 output instead of
        all zeros.

        ``full_xmem_row=1``: always process all 128 lanes.
        ``full_xmem_row=0``: process only ``CR15.valid_elements`` lanes (clamped to 128).

        Requires INT8 mode.

        In wide-vector debug mode, ``aaq`` is normally a no-op (use
        ``STR_ACC_REG`` to dump ``r_acc``). Set ``state.wide_vector_quantize_output``
        to quantize from ``post_aaq_reg`` wide lanes for comparison with the real path.
        """
        dtype = self.state.dtype
        if dtype != DType.INT8:
            raise EmulatorError("AAQ instruction requires INT8 mode")

        active = 128 if full_xmem_row else self._agg_active_lane_count(
            self.state.get_config_valid_elements()
        )

        if self._wide_vector_active():
            if not self.state.wide_vector_quantize_output:
                return
            src_buf = self.state.regfile.raw("post_aaq_reg")
            result = bytearray(128)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                for i in range(active):
                    val = struct.unpack_from("<f", src_buf, i * 4)[0]
                    clamped = max(-128, min(127, int(round(val))))
                    result[i] = clamped & 0xFF
            else:
                for i in range(active):
                    val = struct.unpack_from("<i", src_buf, i * 4)[0]
                    clamped = max(-128, min(127, val))
                    result[i] = clamped & 0xFF
            self.state.regfile.set_post_aaq_reg(result + bytearray(384))
            return

        src_buf = self.state.regfile.raw("post_aaq_reg")
        result = bytearray(128)
        for i in range(active):
            val = struct.unpack_from("<i", src_buf, i * 4)[0]
            clamped = max(-128, min(127, val))  # direct INT8 clamp (no >>24 truncation)
            result[i] = clamped & 0xFF
        self.state.regfile.set_post_aaq_reg(result + bytearray(384))

    def execute_activate(self, *, activation_fn: int, full_xmem_row: int) -> None:
        """Apply element-wise activation to active lanes.

        Reads each active lane from ``r_acc`` and writes the result into the same
        lane of ``post_aaq_reg`` (512-byte wide staging). ``r_acc`` is not modified.
        Lanes at indices ``>=`` the active lane count in ``post_aaq_reg`` are left
        unchanged.

        ``activation_fn`` is the encoded enum index (0–8) from the instruction word.
        full_xmem_row=1: always use 128 lanes. full_xmem_row=0: lane count from CR15.valid_elements.
        """
        fn_id = int(activation_fn) & REGISTER_WORD_VALUE_MASK
        valid_elements = 128 if full_xmem_row else self.state.get_config_valid_elements()
        active = self._agg_active_lane_count(valid_elements)
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.snapshot.raw("r_acc")
        post_buf = self.state.regfile.raw("post_aaq_reg")

        for i in range(active):
            raw = struct.unpack_from(fmt, acc_buf, i * 4)[0]
            y = apply_activation(
                fn_id,
                float(raw),
                elu_alpha=self.state.elu_alpha,
            )
            if fmt == "<i":
                yi = int(round(y))
                if yi < -2147483648:
                    yi = -2147483648
                elif yi > 2147483647:
                    yi = 2147483647
                struct.pack_into("<i", post_buf, i * 4, yi)
            else:
                struct.pack_into("<f", post_buf, i * 4, float(y))

    def execute_str_post_aaq_reg(self, *, offset: int, base: int) -> None:
        """Store **POST_AAQ_REG** to XMEM, at the active mode's element width.

        The store width tracks the only stage that narrows lanes, ``AAQ``:

        - **INT8 (narrow) mode**: ``AAQ`` has packed the 128 quantized INT8 lanes
          into ``post_aaq_reg[0:128]`` (the trailing 384 bytes are zero). The
          INT8 output is therefore exactly **128 bytes** (1 byte/lane), so only
          those are written. This lets a result round-trip as a 128-byte input
          load (``LDR_MULT_REG``) for a later compute pass. Run ``AAQ`` first.

        - **Wide-vector debug mode** (FP32 / INT32 analysis, e.g. softmax or FP8
          multi-pass): lanes stay 4 bytes wide (``AAQ`` is a no-op) and the full
          **512 bytes** are written, matching the 4-byte/lane element width that
          a wide-mode ``LDR_MULT_REG`` reads back.
        """
        addr = offset + base
        post_aaq = self.state.regfile.raw("post_aaq_reg")
        if self.state.dtype == DType.INT8 and not self._wide_vector_active():
            # INT8 narrow mode: drain only the 128 packed INT8 bytes from AAQ.
            self.state.xmem.write_address(addr, bytes(post_aaq[:128]))
        else:
            # Wide-vector / non-INT8: 4-byte lanes; write the full 512 bytes.
            self.state.xmem.write_address(addr, bytes(post_aaq))

    # -----------------------------------------------------------------------
    # COND Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_beq(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute BEQ: Branch if equal."""
        self.state.program_counter = label if reg1 == reg2 else self.state.program_counter + 1

    def execute_bne(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute BNE: Branch if not equal."""
        self.state.program_counter = label if reg1 != reg2 else self.state.program_counter + 1

    @staticmethod
    def _to_signed_32(value: int) -> int:
        """Interpret an unsigned 32-bit value as signed."""
        if value >= 0x80000000:
            return value - 0x100000000
        return value

    def execute_blt(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute BLT: Branch if less than (signed comparison)."""
        s1 = self._to_signed_32(reg1)
        s2 = self._to_signed_32(reg2)
        self.state.program_counter = label if s1 < s2 else self.state.program_counter + 1

    def execute_bnz(self, *, test_reg: int, base_reg: int, label: int) -> None:
        """Execute BNZ: Branch if not zero."""
        self.state.program_counter = label if test_reg != 0 else self.state.program_counter + 1

    def execute_bz(self, *, test_reg: int, base_reg: int, label: int) -> None:
        """Execute BZ: Branch if zero."""
        self.state.program_counter = label if test_reg == 0 else self.state.program_counter + 1

    def execute_b(self, *, label: int) -> None:
        """Execute B: Unconditional branch."""
        self.state.program_counter = label

    def execute_br(self, *, reg: int) -> None:
        """Execute BR: Branch to register value."""
        self.state.program_counter = reg

    def execute_bkpt(self) -> None:
        """Execute BKPT: Breakpoint (halt execution)."""
        self.state.program_counter = INST_MEM_SIZE  # halt

    # -----------------------------------------------------------------------
    # BREAK Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_break_nop(self) -> BreakResult:
        """Execute BREAK_NOP: No operation."""
        return BreakResult.CONTINUE

    def execute_break(self) -> BreakResult:
        """Execute BREAK: Unconditional break."""
        return BreakResult.BREAK

    def execute_break_ifeq(self, *, reg: int, value: int) -> BreakResult:
        """Execute BREAK.IFEQ: Break if LR register equals immediate."""
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
            slot_type: Slot type ("load", "store", "acc_store", "mult", "acc", "cond", "break")
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

        # Update run statistics
        stats = self.state.stats
        if slot_type == "mult":
            if instruction_name != "MULT_NOP":
                stats.mult_active_cycles += 1
        elif slot_type == "acc":
            if instruction_name != "ACC_NOP":
                stats.acc_active_cycles += 1
        elif slot_type == "load":
            if instruction_name != "LOAD_NOP":
                stats.xmem_reads += 1
        elif slot_type in {"store", "acc_store"}:
            if instruction_name not in {"STORE_NOP", "ACC_STORE_NOP"}:
                stats.xmem_writes += 1

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
        4. Execute load, MULT, ACC, AAQ, store, acc_store, COND from the snapshot
           (load before store; same-cycle load+store: load resolves first)

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
        self.dispatch_instruction("load", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("aaq", inst)
        self.dispatch_instruction("store", inst)
        self.dispatch_instruction("acc_store", inst)
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
        self.dispatch_instruction("load", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("aaq", inst)
        self.dispatch_instruction("store", inst)
        self.dispatch_instruction("acc_store", inst)
        self.dispatch_instruction("cond", inst)
