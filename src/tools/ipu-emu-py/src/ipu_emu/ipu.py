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
from functools import lru_cache
from typing import Any, Callable, NamedTuple

import numpy as np

from ipu_emu.ipu_state import IpuState, INST_MEM_SIZE, WideVectorArithmetic
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_emu.regfile import RegFile
from ipu_emu.errors import EmulatorError
from ipu_emu.ipu_math import ipu_mult, ipu_add, ipu_sub, DType
from ipu_emu.ipu_config import REGISTER_WORD_VALUE_MASK, LR_CR_SCALAR_BITS, PadMode, Partition
from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    SLOT_BINARY_LAYOUT,
    SLOT_UNIONS,
    SLOT_COUNT,
    create_emulator_constants,
)
from ipu_common.acc_stride_enums import (
    get_elements_per_row,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_common.incr_mod_pow2_k import LR_MOD_POW2_K_ENCODED_MAX, LR_MOD_POW2_K_MIN
from ipu_common.reshape_mask import RESHAPE_ELEMENT_COUNT, RESHAPE_MASK_LR_OFFSET
from ipu_common.registers import get_register_sizes, get_mult_stage_map
from ipu_common.activations import apply_activation

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

# Number of vector lanes: 128, mode-invariant. An element is 1 byte in narrow
# mode and 4 bytes in wide-vector debug mode, so LANES must never be confused
# with a byte count — use it for lane loop bounds, mask bit-widths, and
# lane-indexed lists/tuples in both modes. R_REG_SIZE remains a byte count
# (the "r" register's size); it coincides with LANES only in narrow mode.
#
# Derived from r_acc's word_view: r_acc is 128 uint32 lanes regardless of mode
# (the same R_ACC_SIZE // 4 word count already used throughout this file for
# acc-slot addressing), so it — not R_REG_SIZE — is the true source for LANES.
LANES = R_ACC_SIZE // 4

# XMEM row geometry is shared by instruction execution and debugging. Derive
# it from the register schema and the active element representation so there
# is one source of truth for mode-dependent row addressing.
_NARROW_ELEMENT_WIDTH_BYTES = R_REG_SIZE // LANES
_WIDE_ELEMENT_WIDTH_BYTES = struct.calcsize("<I")


def xmem_element_width_bytes(state: IpuState) -> int:
    """Return the byte width of one XMEM element in the active mode."""
    if state.wide_vector_debug:
        return _WIDE_ELEMENT_WIDTH_BYTES
    return _NARROW_ELEMENT_WIDTH_BYTES


def xmem_row_size_bytes(state: IpuState) -> int:
    """Return the byte size of one assembly-addressable XMEM row."""
    return LANES * xmem_element_width_bytes(state)


XMEM_ADDRESSABLE_ROWS = XMEM_SIZE_BYTES // (LANES * _WIDE_ELEMENT_WIDTH_BYTES)

# R_CYCLIC is divided into four 128-byte slots; LDR_CYCLIC_MULT_REG's index
# must land exactly on a slot boundary — no implicit wraparound.
R_CYCLIC_VALID_INDICES = tuple(range(0, R_CYCLIC_SIZE, R_REG_SIZE))

# XMEM is allocated 8 MB always (mode-independent); narrow mode may address
# only the first 2 MB of it (16384 rows of 128 B). Debug mode reaches the
# full 8 MB (16384 rows of 512 B).
NARROW_MAX_ROW = XMEM_ADDRESSABLE_ROWS

# 0..LANES-1, for the rotated Ra window MULT.VE reads.
_LANE_INDEX = np.arange(LANES)

# Whole-row struct formats for the narrow datapath, keyed by the per-lane
# format ``Ipu._acc_agg_lane_fmt`` returns.
_ROW_FMT = {"<f": f"<{LANES}f", "<i": f"<{LANES}i"}


def _pack_lanes_one_by_one(lane_fmt: str, buf: bytearray, values, byte_off: int = 0) -> None:
    """The lane-by-lane store, kept for the failure path.

    ``struct.pack_into`` zeroes a field before it packs it and raises on a
    value the format cannot hold, so a row that fails at lane *k* leaves lanes
    ``0..k-1`` written, lane *k* zeroed and the rest untouched. The whole-row
    stores keep that behaviour by replaying this loop when a row fails: it
    fails at the same lane, leaves the same bytes and raises the same error.
    """
    for i, v in enumerate(values):
        struct.pack_into(lane_fmt, buf, byte_off + 4 * i, v)


def _store_row(buf: bytearray, lane_fmt: str, values) -> None:
    """Write LANES values into *buf* as one row, or fail exactly as lane-by-lane would."""
    try:
        packed = struct.pack(_ROW_FMT[lane_fmt], *values)
    except (OverflowError, struct.error):
        packed = None
    if packed is None:
        _pack_lanes_one_by_one(lane_fmt, buf, values)  # raises at the offending lane
        return
    buf[: LANES * 4] = packed


@lru_cache(maxsize=512)
def _inactive_lanes(mask_int: int) -> tuple[int, ...]:
    """Lanes a 128-bit multiply mask deactivates, in ascending order.

    A mask that activates every lane (the default R_MASK, and what every
    kernel in the tree runs with) yields an empty tuple, so the caller writes
    no pad values -- exactly what the per-lane scan did, without the scan.
    """
    return tuple(i for i in range(LANES) if not ((mask_int >> i) & 1))


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
    "LrdIdx": "lrd_reg_field",
    "LrIncDecImmediate": "lr_inc_dec_immediate",
    "AddbiImmediate": "addbi_immediate",
    "ElementsInRow": "elements_in_row_field",
    "HorizontalStride": "horizontal_stride_field",
    "VerticalStride": "vertical_stride_field",
    "LrModPow2KImmediate": "lr_mod_pow2_k_immediate",
    "MultMaskOffsetImmediate": "mult_mask_offset_immediate",
    "ActivationFn": "activation_fn_field",
    "BreakImmediate": "break_immediate_type",
    "Label": "label_token",
    "DstructureCrIdx": "dstructure_cr_reg_field",
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


# ---------------------------------------------------------------------------
# Dispatch plans: everything about an instruction that does not depend on the
# instruction *word*, resolved once at import time.
#
# Dispatch runs nine times per emulated cycle, so anything it recomputes is
# multiplied by nine per cycle: building the opcode field name with an f-string,
# looking the instruction up by position in a freshly built list of keys,
# fetching the handler with getattr, and choosing a stats counter by string
# comparison. All of that is a function of (slot_type, opcode) alone.
# ---------------------------------------------------------------------------

# How a "read" operand resolves to a register value. The three common cases are
# handled inline in the dispatcher; anything else falls back to
# ``Ipu._resolve_operand``.
_READ_LR = 0
_READ_CR = 1
_READ_LCR = 2
_READ_OTHER = 3

_READ_CODES = {"LrIdx": _READ_LR, "CrIdx": _READ_CR, "LcrIdx": _READ_LCR}


class _OperandRead(NamedTuple):
    """One operand that dispatch resolves to a register value."""

    name: str
    op_type: str
    code: int
    from_snapshot: bool


class _Plan(NamedTuple):
    """Everything dispatch needs for one (slot, opcode), precomputed."""

    inst_name: str
    fn: Callable                                   # unbound Ipu method
    fields: tuple[tuple[str, str], ...]            # (operand name, inst field key)
    reads: tuple[_OperandRead, ...]
    stat: str | None                               # RunStats counter to bump
    write_operand: str | None                      # LR slot: name of the target
    write_is_lrd: bool                             # ...and whether it is an LRD pair


# Which RunStats counter a non-NOP instruction in each slot bumps.
_SLOT_STAT = {
    "mult": "mult_active_cycles",
    "acc": "acc_active_cycles",
    "load": "xmem_reads",
    "store": "xmem_writes",
    "acc_store": "xmem_writes",
}


def _build_plan(slot_type: str, inst_name: str, spec: dict, field_map: dict) -> _Plan:
    reads = tuple(
        _OperandRead(
            name=name,
            op_type=op_type,
            code=_READ_CODES.get(op_type, _READ_OTHER),
            from_snapshot=(source == "snapshot"),
        )
        for name, (op_type, source) in _INSTRUCTION_READ_TYPES.get(
            (slot_type, inst_name), {}
        ).items()
    )
    # The LR slot needs the write target to detect two sub-instructions writing
    # the same register; which operand carries it is fixed per instruction.
    write_operand, write_is_lrd = None, False
    if slot_type == "lr":
        for op in spec["operands"]:
            if op["name"] in ("dest", "reg") and "read" not in op:
                write_operand, write_is_lrd = op["name"], op["type"] == "LrdIdx"
                break
    try:
        fn = getattr(Ipu, spec["execute_fn"])
    except AttributeError:
        raise RuntimeError(
            f"instruction_spec {slot_type}/{inst_name} names handler "
            f"Ipu.{spec['execute_fn']}, which does not exist "
            "(see CLAUDE.md, 'Adding an Instruction')"
        ) from None
    return _Plan(
        inst_name=inst_name,
        fn=fn,
        fields=tuple(field_map.items()),
        reads=reads,
        stat=None if inst_name == "NOP" else _SLOT_STAT.get(slot_type),
        write_operand=write_operand,
        write_is_lrd=write_is_lrd,
    )


def _build_slot_plans():
    """Build the opcode-indexed plan tables. Called once, after ``Ipu`` exists.

    Returns ``(opcode_fields, plans, lr_plans, lr_opcode_fields)``.
    """
    opcode_fields = {
        slot_type: f"{prefix}_token_0_{slot_type}_inst_opcode"
        for slot_type, prefix in _SLOT_FIELD_PREFIX.items()
    }
    plans = {
        slot_type: tuple(
            _build_plan(
                slot_type, inst_name, spec, _INSTRUCTION_FIELD_MAP[(slot_type, inst_name)]
            )
            for inst_name, spec in INSTRUCTION_SPEC[slot_type].items()
        )
        for slot_type in _SLOT_FIELD_PREFIX
    }
    # LR sub-slots differ only in which fields of the word they read.
    lr_plans = tuple(
        tuple(
            _build_plan(
                "lr", inst_name, spec, _INSTRUCTION_FIELD_MAP[("lr", inst_name, slot_idx)]
            )
            for inst_name, spec in INSTRUCTION_SPEC["lr"].items()
        )
        for slot_idx in range(SLOT_COUNT["lr"])
    )
    lr_opcode_fields = tuple(
        f"lr_inst_{slot_idx}_token_0_lr_inst_opcode"
        for slot_idx in range(SLOT_COUNT["lr"])
    )
    return opcode_fields, plans, lr_plans, lr_opcode_fields


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
        # Snapshot buffer reused across cycles. Every register is refreshed from
        # the live file at the start of each cycle, so the snapshot is exactly
        # what a fresh copy would hold; reusing the object just avoids
        # reallocating ~5.7 KB of bytearrays
        # per cycle. Safe because no handler ever writes to the snapshot.
        self._snapshot_buffer: RegFile | None = None
        self._snapshot_source: RegFile | None = None  # the file the buffer copies
        # Public/debug snapshot, materialized only when inspected. It is never
        # recycled, so a debugger may retain it across later cycles just as it
        # could before the execution buffer was introduced.
        self._public_snapshot: RegFile | None = None

    @property
    def snapshot(self) -> RegFile | None:
        """Inspectable, cycle-stable copy of the current execution snapshot.

        Built-in handlers read ``_snapshot_buffer`` directly so normal execution
        does not allocate here. Debuggers pay for one copy only if they inspect
        this property, and repeated reads during the same cycle return that same
        independent object.
        """
        if self._snapshot_buffer is None:
            return None
        if self._public_snapshot is None:
            self._public_snapshot = self._snapshot_buffer.snapshot()
        return self._public_snapshot

    def _take_snapshot(self) -> None:
        """Refresh the private execution snapshot at cycle start."""
        regfile = self.state.regfile
        if regfile is not self._snapshot_source:
            # First cycle, or the state swapped in another register file:
            # take a real copy of this one and reuse it from here on.
            self._snapshot_buffer = regfile.snapshot()
            self._snapshot_source = regfile
        else:
            regfile.copy_into(self._snapshot_buffer)
        # A previously exposed snapshot must remain frozen for any debugger
        # retaining it. The next public access lazily copies this cycle.
        self._public_snapshot = None

    # -----------------------------------------------------------------------
    # Wide-vector debug mode (emulator-only; GitHub issue #33)
    # -----------------------------------------------------------------------

    def _wide_vector_active(self) -> bool:
        return self.state.wide_vector_debug

    def _element_width_bytes(self) -> int:
        """Bytes per element in the active mode: 1 narrow, 4 wide-vector debug.

        The single primitive the two modes differ by. Every other
        mode-dependent size (row size, buffer lengths) is derived from this.
        """
        return xmem_element_width_bytes(self.state)

    def _row_size_bytes(self) -> int:
        """Bytes per row (LANES elements) in the active mode: 128 narrow, 512 debug."""
        return xmem_row_size_bytes(self.state)

    def _xmem_row_addr(self, row: int) -> int:
        """Translate an XMEM row number to a byte address in the active mode.

        ``.asm`` XMEM operands (``offset + base``) are row numbers, not byte
        addresses — one row is LANES elements, so the same row number reaches
        the same logical row in both modes at different byte offsets. XMEM is
        allocated 8 MB unconditionally; narrow mode may only *address* the
        first 16384 rows (the first 2 MB) of that allocation.

        This only translates and range-checks the row itself; the resulting
        address's actual payload (which may span more than one row's worth of
        bytes, e.g. STR_ACC_REG's fixed 512-byte R_ACC) is bounds-checked by
        ``XMem.read_address``/``write_address`` against the 8 MB allocation.
        """
        if row < 0:
            raise EmulatorError(f"XMEM row must be non-negative; got {row}")
        if not self._wide_vector_active() and row >= NARROW_MAX_ROW:
            raise EmulatorError(
                f"XMEM row {row} is out of range for narrow mode "
                f"(rows 0..{NARROW_MAX_ROW - 1}, the first 2 MB of the 8 MB allocation)"
            )
        addr = row * self._row_size_bytes()
        if addr >= XMEM_SIZE_BYTES:
            raise EmulatorError(
                f"XMEM row {row} is out of range: byte address {addr} exceeds "
                f"the {XMEM_SIZE_BYTES}-byte allocation"
            )
        return addr

    def _wide_assert_lane_aligned_byte_offset(self, name: str, byte_off: int) -> None:
        """Wide-vector mode treats r_cyclic in 4-byte elements; misaligned offsets corrupt unpacking."""
        if byte_off % 4 != 0:
            raise EmulatorError(
                f"Wide-vector debug: {name} must be 4-byte aligned, got {byte_off}"
            )

    def _wide_imult32(self, a: int, b: int) -> int:
        p = int(a) * int(b)
        p &= 0xFFFFFFFF
        return p - 0x100000000 if p >= 0x80000000 else p

    def _wide_cr_scalar_byte_as_int32(self, cr_idx: int) -> int:
        """Low byte of CR as signed int32 element (CR itself is not widened)."""
        b = self.state.regfile.get_cr(cr_idx) & 0xFF
        return b if b < 128 else b - 256

    # -- wide-vector lanes as arrays ----------------------------------------
    #
    # The scalar loops these replace did their arithmetic on Python floats
    # (doubles) and Python ints, one lane at a time, and rounded once when
    # packing the result back. Widening to float64/int64 here keeps every
    # intermediate identical and leaves that single rounding on the store, so
    # the vectorised path is bit-for-bit the scalar path -- just not 128 round
    # trips through ``struct``.

    def _wide_lanes_from(self, buf, byte_off: int = 0) -> np.ndarray:
        """Read LANES lanes of *buf* widened to float64 (FP32) or int64 (INT32)."""
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            # NumPy's global error policy belongs to the embedding application;
            # signaling-NaN conversion must behave like struct regardless of it.
            with np.errstate(all="ignore"):
                return np.frombuffer(
                    buf, dtype="<f4", count=LANES, offset=byte_off
                ).astype(np.float64)
        return np.frombuffer(
            buf, dtype="<i4", count=LANES, offset=byte_off
        ).astype(np.int64)

    def _wide_store_lanes(self, buf: bytearray, values: np.ndarray, byte_off: int = 0) -> None:
        """Write LANES lanes back, rounding once — what ``pack_into`` did."""
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            # numpy 2 refuses to re-enter one errstate object, so this stays
            # per call (~2 us); a magnitude pre-check measured slower.
            with np.errstate(all="ignore"):
                narrowed = values.astype(np.float32)
            if not np.isfinite(narrowed).all() and np.any(
                np.isinf(narrowed) & np.isfinite(values)
            ):
                # A finite double too large for float32: struct.pack_into
                # raises OverflowError where numpy would store an infinity.
                # Replay the lane-by-lane store so the register ends up exactly
                # as that path left it and the same exception propagates.
                _pack_lanes_one_by_one("<f", buf, values.tolist(), byte_off)
                return
            raw = narrowed
        else:
            # int32 wrap, as _wide_imult32 / ipu_add(INT8) do lane by lane.
            raw = (values & 0xFFFFFFFF).astype(np.uint32)
        buf[byte_off:byte_off + LANES * 4] = raw.tobytes()

    def _wide_ra_lanes(self, mult_stage_enc: int) -> np.ndarray:
        """R0/R1 wide lanes from the cycle-start snapshot (issue #157).

        Read straight from the snapshot's storage: R0 and R1 are the two
        512-byte elements of ``r_wide_debug``, and ``_wide_lanes_from`` makes
        its own widened copy, so nothing aliases the register.
        """
        return self._wide_lanes_from(
            self._snapshot_buffer.raw("r_wide_debug"), mult_stage_enc * LANES * 4
        )

    def _wide_ra_lane(self, idx: int) -> float | int:
        """One lane of the R0 ++ R1 pair (``0 <= idx < 2*LANES``) from the snapshot.

        The pair is contiguous in ``r_wide_debug`` (R0 first), so lane *idx*
        is the 4-byte element at ``4 * idx`` whichever register it falls in.
        """
        fmt = "<f" if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<i"
        return struct.unpack_from(fmt, self._snapshot_buffer.raw("r_wide_debug"), 4 * idx)[0]

    def _wide_rb_lanes(self, cyclic_byte_off: int) -> np.ndarray:
        """r_cyclic wide lanes from the cycle-start snapshot."""
        self._wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_byte_off)
        return self._wide_lanes_from(
            self._snapshot_buffer.get_r_cyclic_wide_debug_at(
                cyclic_byte_off, self._row_size_bytes()
            )
        )

    def _rc_element_to_byte_offset(self, rc_idx: int) -> int:
        """Scale an r_cyclic ELEMENT index to a byte offset (mode-dependent width).

        ``rc_idx`` (MULT.RC.* operand) addresses r_cyclic the same way
        ``LDR_CYCLIC_MULT_REG``'s ``index`` operand does -- in elements, not
        bytes -- so reads and writes to the same register share one unit. A
        no-op in narrow mode (1 byte/element); scales by 4 in wide-vector
        debug mode. See issue #182's follow-up: rc_idx used to be a raw byte
        offset, which silently diverged from the write-side element index
        once r_cyclic's wide-mode ring grew to 2048 B (512 elements x 4 B).
        """
        return rc_idx * self._element_width_bytes()

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
            if raw_value > 1:
                raise EmulatorError(
                    "Mult-stage operand must encode r0 (0) or r1 (1); "
                    f"got {raw_value}"
                )
            if self._wide_vector_active():
                # Mult handlers read wide lanes via _wide_ra_lanes(raw_value),
                # keyed by MultStageReg encoding index (0=r0, 1=r1) into r_wide_debug.
                return raw_value
            reg_name, elem_idx = _MULT_STAGE_MAP[raw_value]
            return source.get_register_bytes(reg_name, elem_idx)
        else:
            return raw_value

    @staticmethod
    @lru_cache(maxsize=None)
    def _build_partition_vector(num_partitions: int) -> int:
        """Build the left-shift partition vector (0 at the START of each group).

        Used for positive mask_shift indices (+1, +2, +3).
        num_partitions must be in VALID_PARTITION_VALUES.
        num_partitions=0: all-ones — no boundaries, shifts are unconstrained.
        num_partitions=P: P groups of LANES/P elements; bit 0 of each group is 0.
        """
        assert isinstance(num_partitions, Partition), (
            f"partition must be a Partition enum value, got {num_partitions!r}"
        )
        if num_partitions == 0:
            return (1 << LANES) - 1
        step = LANES // num_partitions
        result = 0
        for i in range(LANES):
            if i % step != 0:
                result |= (1 << i)
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def _build_inverse_partition_vector(num_partitions: int) -> int:
        """Build the right-shift partition vector (0 at the END of each group).

        Used for negative mask_shift indices (−1, −2, −3).
        num_partitions must be in VALID_PARTITION_VALUES.
        num_partitions=0: all-ones — no boundaries, shifts are unconstrained.
        num_partitions=P: P groups of LANES/P elements; last bit of each group is 0.
        """
        assert isinstance(num_partitions, Partition), (
            f"partition must be a Partition enum value, got {num_partitions!r}"
        )
        if num_partitions == 0:
            return (1 << LANES) - 1
        step = LANES // num_partitions
        result = 0
        for i in range(LANES):
            if i % step != step - 1:
                result |= (1 << i)
        return result

    def _mult_mask_and_shift(self, mask_idx: int, shift: int, cr_idx: int) -> None:
        """Apply sequential shift-and-AND mask generation, then gate mult_res.

        ``shift`` is interpreted as ``mask_shift_idx`` ∈ [−3, +3] (clamped).
        The slot mask M is taken from R_MASK slot ``mask_idx``, then shifted
        sequentially (one bit per step):
          idx 0  → M (unmodified)
          idx +k → shift left  k times, ANDing with partition_vector after each step
          idx −k → shift right k times, ANDing with inverse_partition_vector after each step

        Two partition vectors, both derived from CR[cr_idx].partition:
          partition_vector         — 0 at the START of each group (used for left shifts)
          inverse_partition_vector — 0 at the END   of each group (used for right shifts)

        Elements in mult_res where the resulting mask bit is 0 are set to
        CR[cr_idx].pad_mode's fill value (ZERO, +inf, or -inf). +inf/-inf
        require a floating-point dtype — they have no INT8 representation.

        Mode-blind: the mask is 128 bits (one bit per element) and MULT_RES is
        128 four-byte elements in both narrow and wide-vector debug mode, so the
        same code drives both.
        """
        # LR registers are LR_CR_SCALAR_BITS wide; sign-extend before clamping
        if shift >= (1 << (LR_CR_SCALAR_BITS - 1)):
            shift = shift - (1 << LR_CR_SCALAR_BITS)
        shift = max(-3, min(3, shift))

        # Extract 128-bit base mask from the selected R_MASK slot
        mask_bytes = self.state.regfile.get_r_mask()
        mask_slot = mask_idx % (LANES // 16)  # 8 slots of 128 bits each
        offset = mask_slot * 16
        _128_BIT_MASK = (1 << LANES) - 1
        base_mask = int.from_bytes(mask_bytes[offset:offset + 16], byteorder="little") & _128_BIT_MASK

        dstructure = self.state.get_dstructure_for(cr_idx)
        num_partitions = dstructure.partition

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

        # Fill mult_res lanes where the mask bit is clear (lane deactivated)
        # with the configured pad value (default: zero). Which lanes those are
        # depends only on the mask, so the walk over the 128 bits is cached
        # rather than repeated on every multiply.
        inactive = _inactive_lanes(mask_int)
        if not inactive:
            return
        pad_bytes = self._mult_pad_lane_bytes(dstructure.pad_mode)
        mult_res = self.state.regfile.raw("mult_res")
        for i in inactive:
            mult_res[i * 4:i * 4 + 4] = pad_bytes

    def _mult_pad_lane_bytes(self, pad_mode: PadMode) -> bytes:
        """Encode the 4-byte MULT_RES fill value for a masked-out element.

        ZERO is representable in both INT8 (int32) and float dtypes.
        POS_INF/NEG_INF only exist in floating-point representations, so
        they are rejected under integer element arithmetic.

        Which field decides that differs by mode: in wide-vector debug mode
        element arithmetic is governed by ``wide_vector_arithmetic``, not by
        ``dtype`` (which stays at its INT8 default unless a caller overrides
        it). Using ``dtype`` here would reject +inf/-inf in debug/FP32, where
        infinity is perfectly representable.
        """
        if pad_mode == PadMode.ZERO:
            return b"\x00\x00\x00\x00"
        if not self._lanes_are_float():
            raise EmulatorError(
                f"dstructure pad_mode {pad_mode.name} requires floating-point lanes; "
                "integer lane arithmetic has no infinity representation"
            )
        value = float("inf") if pad_mode == PadMode.POS_INF else float("-inf")
        return struct.pack("<f", value)

    def _lanes_are_float(self) -> bool:
        """Whether MULT_RES elements hold floats, under whichever mode is active."""
        if self._wide_vector_active():
            return self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32
        return self.state.dtype != DType.INT8

    @staticmethod
    def _lrd_lr_indices(n: int) -> tuple[int, int]:
        """Real LR register indices (lo, hi) backing LrdIdx pair n: LRDn = LR(2n+1):LR(2n)."""
        return 2 * n, 2 * n + 1

    def _get_lrd_bytes(self, n: int, regfile: RegFile) -> bytes:
        """Read LRDn's 8 byte elements from a register file (snapshot or live).

        Elements 0-3 are LR(2n)'s bytes (little-endian), elements 4-7 are LR(2n+1)'s.
        """
        lo_idx, hi_idx = self._lrd_lr_indices(n)
        return regfile.get_lr(lo_idx).to_bytes(4, "little") + regfile.get_lr(hi_idx).to_bytes(
            4, "little"
        )

    def _set_lrd_bytes(self, n: int, lanes: bytes | bytearray) -> None:
        """Write LRDn's 8 byte elements to the live register file (inverse of ``_get_lrd_bytes``)."""
        lo_idx, hi_idx = self._lrd_lr_indices(n)
        self.state.regfile.set_lr(lo_idx, int.from_bytes(lanes[0:4], "little"))
        self.state.regfile.set_lr(hi_idx, int.from_bytes(lanes[4:8], "little"))

    # -----------------------------------------------------------------------
    # Memory slot instruction handlers (load / store / acc_store)
    # -----------------------------------------------------------------------

    def execute_load_nop(self) -> None:
        """Execute NOP in load slot: No operation."""
        pass

    def execute_store_nop(self) -> None:
        """Execute NOP in store slot: No operation."""
        pass

    def execute_acc_store_nop(self) -> None:
        """Execute NOP in acc_store slot: No operation."""
        pass

    def execute_str_acc_reg(self, *, offset: int, base: int) -> None:
        """Execute STR_ACC_REG: Store accumulator to memory (debug only).

        Stores all 512 bytes of R_ACC (128 elements x 32-bit accumulator
        width) unconditionally in both modes -- R_ACC's width does not scale
        with the active element width, so "all of R_ACC" is already the
        mode-blind statement. Only the row address is translated; this spans
        4 rows in narrow mode and 1 row in debug mode, which is accepted
        since this is a debug-only instruction by nature (see warning above).
        """
        warnings.warn(
            "[DEBUG ONLY] STR_ACC_REG is not a hardware instruction and is available "
            "for emulator debugging purposes only",
            stacklevel=2,
        )
        addr = self._xmem_row_addr(offset + base)
        acc_data = self.state.regfile.get_r_acc_bytes()
        self.state.xmem.write_address(addr, acc_data)

    def execute_ldr_mult_reg(self, *, dest: int, offset: int, base: int) -> None:
        """Execute LDR_MULT_REG: Load data from memory into a mult stage register."""
        if dest not in (0, 1):
            raise EmulatorError(
                f"LDR_MULT_REG: dest must be 0 (r0) or 1 (r1); got {dest}"
            )
        addr = self._xmem_row_addr(offset + base)
        if self._wide_vector_active():
            data = self.state.xmem.read_address(addr, self._row_size_bytes())
            self.state.regfile.set_r_wide_debug(dest, data)
            return

        data = self.state.xmem.read_address(addr, R_REG_SIZE)
        reg_name, elem_idx = _MULT_STAGE_MAP[dest]
        self.state.regfile.set_register_bytes(reg_name, elem_idx, data)

    def execute_ldr_cyclic_mult_reg(self, *, offset: int, base: int, index: int) -> None:
        """Execute LDR_CYCLIC_MULT_REG: Load with cyclic addressing into r_cyclic.

        ``index`` is an ELEMENT index into r_cyclic's 512-element ring and must
        land on one of the four slot boundaries (0/128/256/384) in both modes
        -- writes only ever replace a whole slot, never a partial one. This is
        unlike reads (MULT.RC.* via ``_wide_rb_lanes``/``get_r_cyclic_wide_debug_at``),
        which are allowed at any element index and may cross a slot boundary.
        """
        addr = self._xmem_row_addr(offset + base)
        if index not in R_CYCLIC_VALID_INDICES:
            raise EmulatorError(
                f"LDR_CYCLIC_MULT_REG: index must be one of {R_CYCLIC_VALID_INDICES} "
                f"(R_CYCLIC slot boundaries); got {index}"
            )
        byte_idx = index * self._element_width_bytes()
        data = self.state.xmem.read_address(addr, self._row_size_bytes())
        if self._wide_vector_active():
            self.state.regfile.set_r_cyclic_wide_debug_at(byte_idx, data)
        else:
            self.state.regfile.set_r_cyclic_at(byte_idx, data)

    def execute_ldr_mult_mask_reg(self, *, offset: int, base: int) -> None:
        """Execute LDR_MULT_MASK_REG: Load mask data from memory.

        Reads only the START of the row -- 128 bytes (8 x 128-bit slots) in
        both modes. The mask is 1 bit per element and does not scale with
        element width, so only the row address is translated; the read size
        stays R_REG_SIZE regardless of mode.
        """
        addr = self._xmem_row_addr(offset + base)
        data = self.state.xmem.read_address(addr, R_REG_SIZE)
        self.state.regfile.set_r_mask(data)

    # -----------------------------------------------------------------------
    # LR Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_lr_set(self, *, reg: int, src: int) -> None:
        """Execute SET: Copy a 32-bit value from a configuration register into an LR."""
        self.state.regfile.set_lr(reg, src & 0xFFFFFFFF)

    def execute_lr_add(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute ADD: uint32 ``dest = src_a + src_b``."""
        self.state.regfile.set_lr(dest, (src_a + src_b) & 0xFFFFFFFF)

    def execute_lr_sub(self, *, dest: int, src_a: int, src_b: int) -> None:
        """Execute SUB: uint32 ``dest = src_a - src_b``."""
        self.state.regfile.set_lr(dest, (src_a - src_b) & 0xFFFFFFFF)

    def execute_lr_inc(self, *, dest: int, imm: int) -> None:
        """Execute INC: uint32 ``dest = dest + imm`` (read-modify-write)."""
        assert self._snapshot_buffer is not None
        cur = self._snapshot_buffer.get_lr(dest)
        self.state.regfile.set_lr(dest, (cur + imm) & 0xFFFFFFFF)

    def execute_lr_dec(self, *, dest: int, imm: int) -> None:
        """Execute DEC: uint32 ``dest = dest - imm`` (read-modify-write)."""
        assert self._snapshot_buffer is not None
        cur = self._snapshot_buffer.get_lr(dest)
        self.state.regfile.set_lr(dest, (cur - imm) & 0xFFFFFFFF)

    def _addb_broadcast(self, dest: int, byte_val: int) -> None:
        """Broadcast-add a signed byte to all 8 elements of LRDn = LR(2n+1):LR(2n), clamped to [0, 255]."""
        assert self._snapshot_buffer is not None
        lanes = bytearray(self._get_lrd_bytes(dest, self._snapshot_buffer))
        signed = byte_val - 256 if byte_val >= 128 else byte_val
        for i in range(8):
            lanes[i] = max(0, min(255, lanes[i] + signed))
        self._set_lrd_bytes(dest, lanes)

    def execute_addb(self, *, dest: int, src_b: int) -> None:
        """Execute ADDB: broadcast-add an LR/CR's low byte (signed) to LRDn's 8 byte elements."""
        self._addb_broadcast(dest, src_b & 0xFF)

    def execute_addbi(self, *, dest: int, imm: int) -> None:
        """Execute ADDBI: broadcast-add an immediate byte (signed) to LRDn's 8 byte elements."""
        self._addb_broadcast(dest, imm)

    def execute_lr_incr_mod_pow2(self, *, dest: int, step: int, k: int) -> None:
        """INCR_MOD_POW2: dest <- (dest + step) mod 2^k.

        Old dest is taken from the cycle-start snapshot (read-before-write). Step is
        resolved from LcrIdx (snapshot), interpreted as uint32 like ``add``/``sub``.
        ``k`` is the raw encoded field (k_semantic − 1); semantic exponent is k + 1.
        """
        assert self._snapshot_buffer is not None
        if k > LR_MOD_POW2_K_ENCODED_MAX:
            raise EmulatorError(
                f"INCR_MOD_POW2: invalid k encoding {k} (max {LR_MOD_POW2_K_ENCODED_MAX})"
            )
        k_exp = k + LR_MOD_POW2_K_MIN
        cur = self._snapshot_buffer.get_lr(dest)
        step_u = step & 0xFFFFFFFF
        mask = (1 << k_exp) - 1
        self.state.regfile.set_lr(dest, ((cur + step_u) & 0xFFFFFFFF) & mask)

    def execute_lr_nop(self) -> None:
        """Execute NOP in LR slot: No operation."""
        pass

    def _dispatch_lr_slots(self, inst: dict[str, int]) -> None:
        """Dispatch all LR sub-slots with conflict detection.

        LR is special: the VLIW word contains multiple LR sub-instructions
        (lr_inst_0, lr_inst_1, …). Each is dispatched independently
        with named operands. Read operands are auto-resolved to values.
        """
        pending: list[tuple[Callable, dict[str, int]]] = []
        all_targets: list[int] = []

        for slot_idx, slot_plans in enumerate(_LR_PLANS):
            plan = slot_plans[inst[_LR_OPCODE_FIELDS[slot_idx]]]
            kwargs = {name: inst[field_key] for name, field_key in plan.fields}
            self._resolve_reads(plan, kwargs)

            # Which real LR indices this sub-instruction writes (an LrdIdx
            # target expands to the two LRs of the pair).
            if plan.write_operand is not None:
                raw = kwargs[plan.write_operand]
                if plan.write_is_lrd:
                    all_targets.extend(Ipu._lrd_lr_indices(raw))
                else:
                    all_targets.append(raw)
            pending.append((plan.fn, kwargs))

        # Conflict check: no two valid instructions may write to the same LR.
        if len(all_targets) != len(set(all_targets)):
            raise RuntimeError(
                f"LR conflict: multiple writes to the same LR register in same cycle "
                f"(targets: {all_targets})"
            )

        for fn, kwargs in pending:
            fn(self, **kwargs)

    # -----------------------------------------------------------------------
    # MULT Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_mult_nop(self) -> None:
        """Execute NOP in mult slot: No operation."""
        pass

    def _mult_resolve_lcr_scalar(self, src: int) -> int:
        """Resolve an LcrIdx ``src`` field that addresses a *byte* scalar.

        If ``src`` encodes an LR, the LR's stored value is itself used as an
        index (mod 256) into the combined Ra buffer (``R0`` ++ ``R1``). If
        ``src`` encodes a CR, the CR's low byte is the scalar directly.
        """
        if src < LR_REG_COUNT:
            # The LR holds an INDEX into Ra -> read it LIVE (same-cycle LR writes
            # are visible, like other mult index operands).  Only the Ra DATA is
            # snapshot (issue #157): a same-cycle LDR_MULT_REG is not yet visible.
            idx = self.state.regfile.get_lr(src) % (2 * R_REG_SIZE)
            r_buf = self._snapshot_buffer.raw("r")  # Ra (R0/R1) DATA from snapshot (issue #157)
            return r_buf[idx]
        cr_idx = src - LR_REG_COUNT
        return self.state.regfile.get_cr(cr_idx) & 0xFF

    def _mult_resolve_lcr_scalar_wide(self, src: int) -> float | int:
        """Wide-vector counterpart of ``_mult_resolve_lcr_scalar``."""
        if src < LR_REG_COUNT:
            # Ra/Rc are read from the START-OF-CYCLE SNAPSHOT like every other
            # register, so a same-cycle LDR_MULT_REG is not visible to the
            # consuming mult (issue #157: the load lands a cycle later).
            return self._wide_ra_lane(self.state.regfile.get_lr(src) % (2 * LANES))
        cr_idx = src - LR_REG_COUNT
        cr_scalar = self._wide_cr_scalar_byte_as_int32(cr_idx)
        if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            return float(cr_scalar)
        return cr_scalar

    def execute_mult_rc_vv(self, *, rc_idx: int, ra: bytearray | int,
                           mask_offset: int, mask_shift: int, cr_idx: int) -> None:
        """Execute MULT.RC.VV: R_CYCLIC vector × Ra (R0/R1) vector, element-wise."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            byte_off = self._rc_element_to_byte_offset(rc_idx)
            rb_lanes = self._wide_rb_lanes(byte_off)
            ra_lanes = self._wide_ra_lanes(ra)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    product = rb_lanes * ra_lanes
            else:
                product = rb_lanes * ra_lanes
            self._wide_store_lanes(mult_res, product)
            self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)
            return

        dtype = self.state.dtype
        rc = self._snapshot_buffer.get_r_cyclic_at(
            self._rc_element_to_byte_offset(rc_idx), R_REG_SIZE
        )

        for i in range(LANES):
            result = ipu_mult(rc[i], ra[i], dtype)
            struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)

    def execute_mult_rc_ve(self, *, rc_idx: int, src: int,
                           mask_offset: int, mask_shift: int, cr_idx: int) -> None:
        """Execute MULT.RC.VE: R_CYCLIC vector × scalar (R0/R1 element or CR value)."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            byte_off = self._rc_element_to_byte_offset(rc_idx)
            scalar = self._mult_resolve_lcr_scalar_wide(src)
            rb_lanes = self._wide_rb_lanes(byte_off)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    product = rb_lanes * float(scalar)
            else:
                product = rb_lanes * int(scalar)
            self._wide_store_lanes(mult_res, product)
            self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)
            return

        dtype = self.state.dtype
        scalar_byte = self._mult_resolve_lcr_scalar(src)
        rc = self._snapshot_buffer.get_r_cyclic_at(
            self._rc_element_to_byte_offset(rc_idx), R_REG_SIZE
        )
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(LANES):
            result = ipu_mult(rc[i], scalar_byte, dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)

    def execute_mult_rc_vs(self, *, rc_idx: int,
                           mask_offset: int, mask_shift: int, cr_idx: int) -> None:
        """Execute MULT.RC.VS: R_CYCLIC vector self-multiply (square), element-wise."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            byte_off = self._rc_element_to_byte_offset(rc_idx)
            rb_lanes = self._wide_rb_lanes(byte_off)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    product = rb_lanes * rb_lanes
            else:
                product = rb_lanes * rb_lanes
            self._wide_store_lanes(mult_res, product)
            self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)
            return

        dtype = self.state.dtype
        rc = self._snapshot_buffer.get_r_cyclic_at(
            self._rc_element_to_byte_offset(rc_idx), R_REG_SIZE
        )
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(LANES):
            result = ipu_mult(rc[i], rc[i], dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift, cr_idx)

    def execute_mult_ve(self, *, ra_idx: int, cr_idx: int,
                        mask_offset: int, mask_shift: int, dstructure_cr_idx: int) -> None:
        """Execute MULT.VE: Ra (combined R0/R1) vector × CR scalar, element-wise."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            cr_scalar = self._wide_cr_scalar_byte_as_int32(cr_idx)
            # Ra is the R0 ++ R1 pair read as one 256-lane cyclic window.
            combined = np.concatenate((self._wide_ra_lanes(0), self._wide_ra_lanes(1)))
            window = combined[(ra_idx + _LANE_INDEX) % (2 * LANES)]
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    product = window * float(cr_scalar)
            else:
                product = window * int(cr_scalar)
            self._wide_store_lanes(mult_res, product)
            self._mult_mask_and_shift(mask_offset, mask_shift, dstructure_cr_idx)
            return

        dtype = self.state.dtype
        scalar_byte = self.state.regfile.get_cr(cr_idx) & 0xFF
        r_buf = self._snapshot_buffer.raw("r")  # Ra (R0/R1) from snapshot (issue #157); [0:128]=r0, [128:256]=r1
        fmt = "<i" if dtype == DType.INT8 else "<f"

        for i in range(LANES):
            pos = (ra_idx + i) % (2 * R_REG_SIZE)
            result = ipu_mult(r_buf[pos], scalar_byte, dtype)
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift, dstructure_cr_idx)

    def execute_mult_ee(self, *, ra_idx: int, cr_idx: int,
                        mask_offset: int, mask_shift: int, dstructure_cr_idx: int) -> None:
        """Execute MULT.EE: single Ra element × CR scalar, broadcast to all 128 elements."""
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            cr_scalar = self._wide_cr_scalar_byte_as_int32(cr_idx)
            ra_lane = self._wide_ra_lane(ra_idx % (2 * LANES))
            # The product is one scalar operation, broadcast to every lane.
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                result = float(ra_lane) * float(cr_scalar)
            else:
                result = self._wide_imult32(int(ra_lane), cr_scalar)
            self._wide_store_lanes(mult_res, np.full(LANES, result))
            self._mult_mask_and_shift(mask_offset, mask_shift, dstructure_cr_idx)
            return

        dtype = self.state.dtype
        scalar_byte = self.state.regfile.get_cr(cr_idx) & 0xFF
        r_buf = self._snapshot_buffer.raw("r")  # Ra (R0/R1) from snapshot (issue #157); [0:128]=r0, [128:256]=r1
        fmt = "<i" if dtype == DType.INT8 else "<f"

        ra_byte = r_buf[ra_idx % (2 * R_REG_SIZE)]
        result = ipu_mult(ra_byte, scalar_byte, dtype)
        for i in range(LANES):
            struct.pack_into(fmt, mult_res, i * 4, result)

        self._mult_mask_and_shift(mask_offset, mask_shift, dstructure_cr_idx)

    # -----------------------------------------------------------------------
    # ACC Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_acc_nop(self) -> None:
        """Execute NOP in acc slot: No operation."""
        pass

    def _store_mult_res_row_in_acc(self) -> None:
        """Store MULT_RES in R_ACC through the parent lane conversion path."""
        fmt = self._acc_agg_lane_fmt()
        mult_res = self.state.regfile.raw("mult_res")
        values = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
        _store_row(self.state.regfile.raw("r_acc"), fmt, values)

    def execute_acc_add(self) -> None:
        """Execute ACC.ADD: Accumulate mult_res into accumulator (running add)."""
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self._snapshot_buffer.raw("r_acc")

        if self._wide_vector_active():
            acc_lanes = self._wide_lanes_from(snap_acc)
            mult_lanes = self._wide_lanes_from(mult_res)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    result = acc_lanes + mult_lanes
            else:
                result = acc_lanes + mult_lanes
            self._wide_store_lanes(acc_buf, result)
            return

        dtype = self.state.dtype
        fmt = self._acc_agg_lane_fmt()
        acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
        mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
        _store_row(acc_buf, fmt, [ipu_add(a, m, dtype) for a, m in zip(acc_vals, mult_vals)])

    def execute_acc_add_first(self) -> None:
        """Execute ACC.ADD.FIRST: Set r_acc to multiply result (no previous sum)."""
        self._store_mult_res_row_in_acc()

    def execute_acc_max(self) -> None:
        """Execute ACC.MAX: Each R_ACC element takes max(R_ACC[i], MULT_RES[i])."""
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self._snapshot_buffer.raw("r_acc")

        if self._wide_vector_active():
            acc_lanes = self._wide_lanes_from(snap_acc)
            mult_lanes = self._wide_lanes_from(mult_res)
            # max(a, b) returns b only when b > a -- the NaN-carrying case
            # differs from np.maximum, so the comparison is written out.
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    result = np.where(mult_lanes > acc_lanes, mult_lanes, acc_lanes)
            else:
                result = np.where(mult_lanes > acc_lanes, mult_lanes, acc_lanes)
            self._wide_store_lanes(acc_buf, result)
            return

        fmt = self._acc_agg_lane_fmt()
        acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
        mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
        _store_row(acc_buf, fmt, [max(a, m) for a, m in zip(acc_vals, mult_vals)])

    def execute_acc_max_first(self) -> None:
        """Execute ACC.MAX.FIRST: Overwrite each R_ACC element with MULT_RES (clean init for max)."""
        self._store_mult_res_row_in_acc()

    def execute_acc_sub(self) -> None:
        """Execute ACC.SUB: Subtract MULT_RES from each R_ACC element (running subtract)."""
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")
        snap_acc = self._snapshot_buffer.raw("r_acc")

        if self._wide_vector_active():
            acc_lanes = self._wide_lanes_from(snap_acc)
            mult_lanes = self._wide_lanes_from(mult_res)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    result = acc_lanes - mult_lanes
            else:
                result = acc_lanes - mult_lanes
            self._wide_store_lanes(acc_buf, result)
            return

        dtype = self.state.dtype
        fmt = self._acc_agg_lane_fmt()
        acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
        mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
        _store_row(acc_buf, fmt, [ipu_sub(a, m, dtype) for a, m in zip(acc_vals, mult_vals)])

    def execute_acc_sub_first(self) -> None:
        """Execute ACC.SUB.FIRST: Set each R_ACC element to negated MULT_RES (clean init for subtract)."""
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")

        if self._wide_vector_active():
            # 0 - x, not -x: for x = +0.0 the scalar path produced +0.0, and
            # negation would flip the sign bit.
            mult_lanes = self._wide_lanes_from(mult_res)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                with np.errstate(all="ignore"):
                    result = 0 - mult_lanes
            else:
                result = 0 - mult_lanes
            self._wide_store_lanes(acc_buf, result)
            return

        dtype = self.state.dtype
        fmt = self._acc_agg_lane_fmt()
        zero = 0 if dtype == DType.INT8 else 0.0
        mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
        _store_row(acc_buf, fmt, [ipu_sub(zero, m, dtype) for m in mult_vals])

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
        num_rows = LANES // elements_per_row

        h_enabled, h_inverted = get_horizontal_stride_bits(horizontal_stride)
        v_enabled, v_inverted = get_vertical_stride_bits(vertical_stride)

        # Build list of source indices (0..127) or -1 for zero padding
        after_h: list[int] = []
        if not h_enabled:
            after_h = list(range(LANES))
            effective_row_len = elements_per_row
        else:
            half = elements_per_row // 2
            for row in range(num_rows):
                base = row * elements_per_row
                if h_inverted:
                    indices_in_row = [base + 1 + 2 * j for j in range(half)]
                else:
                    indices_in_row = [base + 2 * j for j in range(half)]
                after_h.extend(indices_in_row)
            effective_row_len = half

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
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.state.regfile.raw("r_acc")
        mult_res = self.state.regfile.raw("mult_res")

        for i, idx in enumerate(out_indices):
            if idx >= 0:
                val = struct.unpack_from(fmt, mult_res, idx * 4)[0]
            else:
                val = 0.0 if fmt == "<f" else 0
            struct.pack_into(fmt, acc_buf, (base + i) * 4, val)

    def execute_acc_reshape(self, *, source: int, dest: int, reshape_mask: int) -> None:
        """Execute ACC.RESHAPE: scatter MULT_RES elements into R_ACC via two LRDn byte-index arrays.

        ``source``/``dest`` are LrdIdx pair indices (0-7); each resolves to 8 byte
        elements read from the pre-instruction snapshot. Only the trailing
        (RESHAPE_ELEMENT_COUNT - mask) elements (indices mask..RESHAPE_ELEMENT_COUNT-1)
        participate, where mask is either the immediate value (encoded 0-7) or
        the value of the specified LR register (encoded >= RESHAPE_MASK_LR_OFFSET,
        LR_index = encoded - RESHAPE_MASK_LR_OFFSET). A resolved mask greater than
        RESHAPE_ELEMENT_COUNT raises rather than being clamped. All MULT_RES reads
        come from the pre-instruction snapshot; a participating source[i]/dest[i]
        outside [0, 127] raises rather than being silently skipped.
        """
        assert self._snapshot_buffer is not None
        src_bytes = self._get_lrd_bytes(source, self._snapshot_buffer)
        dst_bytes = self._get_lrd_bytes(dest, self._snapshot_buffer)

        if reshape_mask >= RESHAPE_MASK_LR_OFFSET:
            lr_idx = reshape_mask - RESHAPE_MASK_LR_OFFSET
            mask = self.state.regfile.get_lr(lr_idx)
        else:
            mask = reshape_mask
        if mask > RESHAPE_ELEMENT_COUNT:
            raise EmulatorError(
                f"ACC.RESHAPE reshape_mask value {mask} exceeds RESHAPE_ELEMENT_COUNT "
                f"({RESHAPE_ELEMENT_COUNT})"
            )

        mult_res = self._snapshot_buffer.raw("mult_res")
        writes = []
        for i in range(mask, RESHAPE_ELEMENT_COUNT):
            if src_bytes[i] >= LANES or dst_bytes[i] >= LANES:
                raise EmulatorError(
                    f"ACC.RESHAPE element {i}: source={src_bytes[i]}, dest={dst_bytes[i]} "
                    f"must both be in [0, {LANES - 1}]"
                )
            writes.append(
                (dst_bytes[i], struct.unpack_from("<I", mult_res, src_bytes[i] * 4)[0])
            )
        for dest_idx, value in writes:
            self.state.regfile.set_r_acc_word(dest_idx, value)

    def execute_aaq_nop(self) -> None:
        """Execute NOP in aaq slot: No operation."""
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
        # Summed in lane order, one lane at a time: floating-point addition is
        # not associative, so a pairwise (numpy) sum would give a different
        # answer. Only the unpacking is batched.
        total: float | int = 0.0 if fmt == "<f" else 0
        for value in struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)[:active]:
            total += value
        return total

    def _agg_max_lanes(
        self, fmt: str, snap_acc: bytearray, active: int, seed: float | int
    ) -> float | int:
        best = seed
        for v in struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)[:active]:
            if v > best:
                best = v
        return best

    def execute_agg_sum_first(self, *, dest_slot: int, cr_idx: int) -> None:
        """Execute AGG.SUM.FIRST: sum active MULT_RES elements, write to R_ACC[dest] (clean init)."""
        valid_elements = self.state.get_dstructure_for(cr_idx).valid_elements
        fmt = self._acc_agg_lane_fmt()
        mult_res = self.state.regfile.raw("mult_res")
        active = self._agg_active_lane_count(valid_elements)
        result = self._agg_sum_lanes(fmt, mult_res, active)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        if fmt == "<i":
            result = self._to_int32(result)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_sum(self, *, dest_slot: int, cr_idx: int) -> None:
        """Execute AGG.SUM: sum active MULT_RES elements and add to R_ACC[dest] (running accumulation)."""
        valid_elements = self.state.get_dstructure_for(cr_idx).valid_elements
        fmt = self._acc_agg_lane_fmt()
        mult_res = self.state.regfile.raw("mult_res")
        active = self._agg_active_lane_count(valid_elements)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        snap_dest = struct.unpack_from(fmt, self._snapshot_buffer.raw("r_acc"), dest * 4)[0]
        partial = self._agg_sum_lanes(fmt, mult_res, active)
        if fmt == "<f":
            result: float | int = float(partial) + float(snap_dest)
        else:
            # fmt == "<i" covers both narrow INT8 mode and wide-vector INT32 mode
            # (_acc_agg_lane_fmt); ipu_add's DType.INT8 branch is plain 32-bit wrap
            # with no 8-bit saturation, so it's correct for INT32 lanes too.
            result = ipu_add(self._to_int32(partial), int(snap_dest), DType.INT8)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_max_first(self, *, dest_slot: int, cr_idx: int) -> None:
        """Execute AGG.MAX.FIRST: max of active MULT_RES elements, write to R_ACC[dest] (no seed).

        When no elements are active (valid_elements=0) the identity seed
        (INT32_MIN / -inf) is written, so the destination is always defined.
        """
        valid_elements = self.state.get_dstructure_for(cr_idx).valid_elements
        fmt = self._acc_agg_lane_fmt()
        mult_res = self.state.regfile.raw("mult_res")
        active = self._agg_active_lane_count(valid_elements)
        seed: float | int = -2147483648 if fmt == "<i" else float("-inf")
        result = self._agg_max_lanes(fmt, mult_res, active, seed)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_agg_max(self, *, dest_slot: int, cr_idx: int) -> None:
        """Execute AGG.MAX: max of active MULT_RES elements seeded with R_ACC[dest] (running max)."""
        valid_elements = self.state.get_dstructure_for(cr_idx).valid_elements
        fmt = self._acc_agg_lane_fmt()
        mult_res = self.state.regfile.raw("mult_res")
        active = self._agg_active_lane_count(valid_elements)
        dest = int(dest_slot) % (R_ACC_SIZE // 4)
        snap_dest = struct.unpack_from(fmt, self._snapshot_buffer.raw("r_acc"), dest * 4)[0]
        result = self._agg_max_lanes(fmt, mult_res, active, snap_dest)
        struct.pack_into(fmt, self.state.regfile.raw("r_acc"), dest * 4, result)

    def execute_activate_quantize(self, *, activation_fn: int, cr_idx: int) -> None:
        """Apply element-wise activation then quantize to INT8.

        Reads each active element from the live ``r_acc`` register file, applies
        the selected activation, clamps to the INT8 range ``[-128, 127]``, and stores
        the resulting bytes in the leading active-element positions of ``post_aaq_reg``;
        all remaining bytes are zeroed. ``r_acc`` is not modified.

        The active element count comes from ``cr_idx``'s decoded dstructure
        ``valid_elements`` field; the caller must name a CR register explicitly.

        Requires INT8 mode in normal operation. In wide-vector debug mode, activation
        is always applied; quantization only occurs when ``state.wide_vector_quantize_output``
        is set (enabling comparison with the real INT8 path).
        """
        fn_id = int(activation_fn) & REGISTER_WORD_VALUE_MASK
        valid_elements = self.state.get_dstructure_for(cr_idx).valid_elements
        active = self._agg_active_lane_count(valid_elements)
        fmt = self._acc_agg_lane_fmt()
        acc_buf = self.state.regfile.raw("r_acc")
        post_buf = self.state.regfile.raw("post_aaq_reg")

        if self._wide_vector_active():
            for i in range(active):
                raw = struct.unpack_from(fmt, acc_buf, i * 4)[0]
                y = apply_activation(
                    fn_id,
                    float(raw),
                    elu_alpha=self.state.elu_alpha,
                    window_a=self.state.window_a,
                    window_b=self.state.window_b,
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

            if not self.state.wide_vector_quantize_output:
                return

            result = bytearray(128)
            if self.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
                for i in range(active):
                    val = struct.unpack_from("<f", post_buf, i * 4)[0]
                    result[i] = max(-128, min(127, int(round(val)))) & 0xFF
            else:
                for i in range(active):
                    val = struct.unpack_from("<i", post_buf, i * 4)[0]
                    result[i] = max(-128, min(127, val)) & 0xFF
            self.state.regfile.set_post_aaq_reg(result + bytearray(384))
            return

        if self.state.dtype != DType.INT8:
            raise EmulatorError("ACTIVATE.QUANTIZE instruction requires INT8 mode")

        result = bytearray(128)
        for i in range(active):
            raw = struct.unpack_from(fmt, acc_buf, i * 4)[0]
            y = apply_activation(
                fn_id,
                float(raw),
                elu_alpha=self.state.elu_alpha,
                window_a=self.state.window_a,
                window_b=self.state.window_b,
            )
            result[i] = max(-128, min(127, int(round(y)))) & 0xFF
        self.state.regfile.set_post_aaq_reg(result + bytearray(384))

    def execute_str_post_aaq_reg(self, *, offset: int, base: int) -> None:
        """Store **POST_AAQ_REG** (512 bytes) to XMEM.

        POST_AAQ_REG's width does not scale with the active element width
        (like R_ACC), so all 512 bytes are written unconditionally in both
        modes; only the row address is translated.
        """
        addr = self._xmem_row_addr(offset + base)
        self.state.xmem.write_address(addr, bytes(self.state.regfile.raw("post_aaq_reg")))

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
    def _to_signed_reg(value: int) -> int:
        """Sign-extend a value at the LR/CR register width (32 bits)."""
        if value >= (1 << (LR_CR_SCALAR_BITS - 1)):
            return value - (1 << LR_CR_SCALAR_BITS)
        return value

    def execute_blt(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute BLT: Branch if less than (signed comparison)."""
        s1 = self._to_signed_reg(reg1)
        s2 = self._to_signed_reg(reg2)
        self.state.program_counter = label if s1 < s2 else self.state.program_counter + 1

    def execute_bge(self, *, reg1: int, reg2: int, label: int) -> None:
        """Execute BGE: Branch if greater or equal (signed comparison)."""
        s1 = self._to_signed_reg(reg1)
        s2 = self._to_signed_reg(reg2)
        self.state.program_counter = label if s1 >= s2 else self.state.program_counter + 1

    def execute_br(self, *, reg: int) -> None:
        """Execute BR: Branch to register value."""
        self.state.program_counter = reg

    def execute_bkpt(self) -> None:
        """Execute BKPT: Breakpoint (halt execution)."""
        self.state.program_counter = INST_MEM_SIZE  # halt

    def execute_cond_nop(self) -> None:
        """Execute NOP in cond slot: No operation; advance PC."""
        self.state.program_counter += 1

    # -----------------------------------------------------------------------
    # BREAK Instruction Handlers
    # -----------------------------------------------------------------------

    def execute_break_nop(self) -> BreakResult:
        """Execute NOP in break slot: No operation."""
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
        plan = _SLOT_PLANS[slot_type][inst[_SLOT_OPCODE_FIELDS[slot_type]]]

        # Extract named operand values
        kwargs = {name: inst[field_key] for name, field_key in plan.fields}

        # Auto-resolve 'read' operands to register values.
        self._resolve_reads(plan, kwargs)

        # Update run statistics
        if plan.stat is not None:
            stats = self.state.stats
            setattr(stats, plan.stat, getattr(stats, plan.stat) + 1)

        # Call handler with named arguments
        return plan.fn(self, **kwargs)

    def _resolve_reads(self, plan: _Plan, kwargs: dict[str, Any]) -> None:
        """Replace each ``read`` operand's raw index with its register value.

        The three scalar index types are inlined because they cover almost every
        read operand in the ISA; anything else goes through
        :meth:`_resolve_operand`.
        """
        for read in plan.reads:
            source = self._snapshot_buffer if read.from_snapshot else self.state.regfile
            raw = kwargs[read.name]
            code = read.code
            if code == _READ_LR:
                kwargs[read.name] = source.get_lr(raw)
            elif code == _READ_CR:
                kwargs[read.name] = source.get_cr(raw)
            elif code == _READ_LCR:
                kwargs[read.name] = (
                    source.get_lr(raw)
                    if raw < LR_REG_COUNT
                    else source.get_cr(raw - LR_REG_COUNT)
                )
            else:
                kwargs[read.name] = self._resolve_operand(read.op_type, raw, source)

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

        self._take_snapshot()

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

        self._take_snapshot()

        # Execute all slots except break
        self._dispatch_lr_slots(inst)
        self.dispatch_instruction("load", inst)
        self.dispatch_instruction("mult", inst)
        self.dispatch_instruction("acc", inst)
        self.dispatch_instruction("aaq", inst)
        self.dispatch_instruction("store", inst)
        self.dispatch_instruction("acc_store", inst)
        self.dispatch_instruction("cond", inst)


# Built here rather than beside the other precomputed tables: a plan holds the
# handler function itself, so ``Ipu`` has to exist first.
_SLOT_OPCODE_FIELDS, _SLOT_PLANS, _LR_PLANS, _LR_OPCODE_FIELDS = _build_slot_plans()
