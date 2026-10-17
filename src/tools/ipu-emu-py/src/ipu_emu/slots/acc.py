"""ACC slot handlers: accumulate (ACC.*), reduce (AGG.*), ACC.STRIDE and ACC.RESHAPE."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.errors import EmulatorError, EmulatorFault
from ipu_emu.geometry import agg_lane_count, lane_format, lrd_bytes, store_wide_lanes, wide_lanes
from ipu_emu.ipu_math import DType, ipu_add, ipu_sub
from ipu_emu.ipu_state import WideVectorArithmetic
from ipu_isa.acc_stride_enums import (
    get_elements_per_row,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_isa.faults import FaultId
from ipu_isa.geometry import LANES
from ipu_isa.reshape_mask import RESHAPE_ELEMENT_COUNT, RESHAPE_MASK_LR_OFFSET

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu

# Whole-row struct formats for the narrow datapath, keyed by the per-lane
# format ``lane_format`` returns.
_ROW_FMT = {"<f": f"<{LANES}f", "<i": f"<{LANES}i"}


def _pack_lanes_one_by_one(
    lane_fmt: str, buf: bytearray | memoryview, values, byte_off: int = 0
) -> None:
    """The lane-by-lane store, kept for the failure path.

    ``struct.pack_into`` zeroes a field before it packs it and raises on a
    value the format cannot hold, so a row that fails at lane *k* leaves lanes
    ``0..k-1`` written, lane *k* zeroed and the rest untouched. The whole-row
    stores keep that behaviour by replaying this loop when a row fails: it
    fails at the same lane, leaves the same bytes and raises the same error.
    """
    for i, v in enumerate(values):
        struct.pack_into(lane_fmt, buf, byte_off + 4 * i, v)


def _store_row(buf: bytearray | memoryview, lane_fmt: str, values) -> None:
    """Write LANES values into *buf* as one row, or fail exactly as lane-by-lane would."""
    try:
        packed = struct.pack(_ROW_FMT[lane_fmt], *values)
    except (OverflowError, struct.error):
        packed = None
    if packed is None:
        _pack_lanes_one_by_one(lane_fmt, buf, values)  # raises at the offending lane
        return
    buf[: LANES * 4] = packed


# ---------------------------------------------------------------------------
# ACC.*: accumulate MULT_RES into R_ACC
# ---------------------------------------------------------------------------


def execute_acc_nop(ipu: Ipu) -> None:
    """Execute NOP in acc slot: No operation."""


def _store_mult_res_row_in_acc(ipu: Ipu) -> None:
    """Store MULT_RES in R_ACC through the parent lane conversion path."""
    fmt = lane_format(ipu.state)
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    values = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
    _store_row(ipu.state.regfile.mutable("r_acc"), fmt, values)


def execute_acc_add(ipu: Ipu) -> None:
    """Execute ACC.ADD: Accumulate mult_res into accumulator (running add)."""
    state = ipu.state
    acc_buf = state.regfile.mutable("r_acc")
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    snap_acc = ipu._view.read_forgetting_provenance("r_acc")

    if state.wide_vector_debug:
        acc_lanes = wide_lanes(state, snap_acc)
        mult_lanes = wide_lanes(state, mult_res)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                result = acc_lanes + mult_lanes
        else:
            result = acc_lanes + mult_lanes
        store_wide_lanes(state, acc_buf, result)
        return

    dtype = state.dtype
    fmt = lane_format(state)
    acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
    mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
    _store_row(
        acc_buf, fmt, [ipu_add(a, m, dtype) for a, m in zip(acc_vals, mult_vals, strict=True)]
    )


def execute_acc_add_first(ipu: Ipu) -> None:
    """Execute ACC.ADD.FIRST: Set r_acc to multiply result (no previous sum)."""
    _store_mult_res_row_in_acc(ipu)


def execute_acc_max(ipu: Ipu) -> None:
    """Execute ACC.MAX: Each R_ACC element takes max(R_ACC[i], MULT_RES[i])."""
    state = ipu.state
    acc_buf = state.regfile.mutable("r_acc")
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    snap_acc = ipu._view.read_forgetting_provenance("r_acc")

    if state.wide_vector_debug:
        acc_lanes = wide_lanes(state, snap_acc)
        mult_lanes = wide_lanes(state, mult_res)
        # max(a, b) returns b only when b > a -- the NaN-carrying case
        # differs from np.maximum, so the comparison is written out.
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                result = np.where(mult_lanes > acc_lanes, mult_lanes, acc_lanes)
        else:
            result = np.where(mult_lanes > acc_lanes, mult_lanes, acc_lanes)
        store_wide_lanes(state, acc_buf, result)
        return

    fmt = lane_format(state)
    acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
    mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
    _store_row(acc_buf, fmt, [max(a, m) for a, m in zip(acc_vals, mult_vals, strict=True)])


def execute_acc_max_first(ipu: Ipu) -> None:
    """Execute ACC.MAX.FIRST: Overwrite each R_ACC element with MULT_RES (clean init for max)."""
    _store_mult_res_row_in_acc(ipu)


def execute_acc_sub(ipu: Ipu) -> None:
    """Execute ACC.SUB: Subtract MULT_RES from each R_ACC element (running subtract)."""
    state = ipu.state
    acc_buf = state.regfile.mutable("r_acc")
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    snap_acc = ipu._view.read_forgetting_provenance("r_acc")

    if state.wide_vector_debug:
        acc_lanes = wide_lanes(state, snap_acc)
        mult_lanes = wide_lanes(state, mult_res)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                result = acc_lanes - mult_lanes
        else:
            result = acc_lanes - mult_lanes
        store_wide_lanes(state, acc_buf, result)
        return

    dtype = state.dtype
    fmt = lane_format(state)
    acc_vals = struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)
    mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
    _store_row(
        acc_buf, fmt, [ipu_sub(a, m, dtype) for a, m in zip(acc_vals, mult_vals, strict=True)]
    )


def execute_acc_sub_first(ipu: Ipu) -> None:
    """Execute ACC.SUB.FIRST: Set each R_ACC element to negated MULT_RES (clean init for subtract)."""
    state = ipu.state
    acc_buf = state.regfile.mutable("r_acc")
    mult_res = ipu._view.read_forgetting_provenance("mult_res")

    if state.wide_vector_debug:
        # 0 - x, not -x: for x = +0.0 the scalar path produced +0.0, and
        # negation would flip the sign bit.
        mult_lanes = wide_lanes(state, mult_res)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                result = 0 - mult_lanes
        else:
            result = 0 - mult_lanes
        store_wide_lanes(state, acc_buf, result)
        return

    dtype = state.dtype
    fmt = lane_format(state)
    zero = 0 if dtype == DType.INT8 else 0.0
    mult_vals = struct.unpack_from(_ROW_FMT[fmt], mult_res, 0)
    _store_row(acc_buf, fmt, [ipu_sub(zero, m, dtype) for m in mult_vals])


# ---------------------------------------------------------------------------
# ACC.STRIDE and ACC.RESHAPE: move MULT_RES lanes into R_ACC
# ---------------------------------------------------------------------------


def _stride_indices(elements_in_row: int, horizontal_stride: int, vertical_stride: int):
    """The MULT_RES lane ACC.STRIDE writes to each R_ACC lane, in order (-1 pads with zero)."""
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
        return after_h
    out_indices = []
    row_sel = range(1, num_rows_after_h, 2) if v_inverted else range(0, num_rows_after_h, 2)
    for r in row_sel:
        start = r * effective_row_len
        out_indices.extend(after_h[start : start + effective_row_len])
    return out_indices


def execute_acc_stride(
    ipu: Ipu,
    *,
    elements_in_row: int,
    horizontal_stride: int,
    vertical_stride: int,
    offset: int,
) -> None:
    """Execute ACC.STRIDE: Decimate mult_res by horizontal/vertical stride and write into r_acc.

    Operand semantics from ipu_isa.acc_stride_enums (single source of truth).
    offset: LR value; (offset % 4) * 32 is the start index in r_acc (0, 32, 64, or 96).
    """
    out_indices = _stride_indices(elements_in_row, horizontal_stride, vertical_stride)

    base = (offset % 4) * 32
    fmt = lane_format(ipu.state)
    # From the first lane written to the end of R_ACC.
    acc_buf = ipu.state.regfile.mutable("r_acc", base * 4)
    if len(out_indices) * 4 > len(acc_buf):
        # The ISA leaves start + N > R_ACC undefined (stage-accumulator 6.8):
        # an open question, not a fault yet, so an EmulatorError before any
        # lane is written.
        raise EmulatorError(
            f"ACC.STRIDE writes {len(out_indices)} elements from lane {base}, "
            f"past the end of R_ACC ({base + len(acc_buf) // 4} lanes)"
        )
    mult_res = ipu._view.read_forgetting_provenance("mult_res")

    for i, idx in enumerate(out_indices):
        if idx >= 0:
            val = struct.unpack_from(fmt, mult_res, idx * 4)[0]
        else:
            val = 0.0 if fmt == "<f" else 0
        struct.pack_into(fmt, acc_buf, i * 4, val)


def execute_acc_reshape(ipu: Ipu, *, source: int, dest: int, reshape_mask: int) -> None:
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
    src_bytes = lrd_bytes(ipu._view.source("lr", "source"), source)
    dst_bytes = lrd_bytes(ipu._view.source("lr", "dest"), dest)

    if reshape_mask >= RESHAPE_MASK_LR_OFFSET:
        lr_idx = reshape_mask - RESHAPE_MASK_LR_OFFSET
        mask = ipu._view.lr(lr_idx, "reshape_mask")
    else:
        mask = reshape_mask
    if mask > RESHAPE_ELEMENT_COUNT:
        raise EmulatorFault(
            FaultId.F5,
            ipu.state.program_counter,
            f"ACC.RESHAPE reshape_mask value {mask} exceeds RESHAPE_ELEMENT_COUNT "
            f"({RESHAPE_ELEMENT_COUNT})",
        )

    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    writes = []
    for i in range(mask, RESHAPE_ELEMENT_COUNT):
        if src_bytes[i] >= LANES or dst_bytes[i] >= LANES:
            raise EmulatorFault(
                FaultId.F5,
                ipu.state.program_counter,
                f"ACC.RESHAPE element {i}: source={src_bytes[i]}, dest={dst_bytes[i]} "
                f"must both be in [0, {LANES - 1}]",
            )
        writes.append((dst_bytes[i], struct.unpack_from("<I", mult_res, src_bytes[i] * 4)[0]))
    for dest_idx, value in writes:
        ipu.state.regfile.set_r_acc_word(dest_idx, value)


# ---------------------------------------------------------------------------
# AGG.*: reduce MULT_RES into one R_ACC element
# ---------------------------------------------------------------------------


def _to_int32(val: int) -> int:
    v = int(val) & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _agg_sum_lanes(fmt: str, snap_acc: bytes | memoryview, active: int) -> float | int:
    # Summed in lane order, one lane at a time: floating-point addition is
    # not associative, so a pairwise (numpy) sum would give a different
    # answer. Only the unpacking is batched.
    total: float | int = 0.0 if fmt == "<f" else 0
    for value in struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)[:active]:
        total += value
    return total


def _agg_max_lanes(
    fmt: str, snap_acc: bytes | memoryview, active: int, seed: float | int
) -> float | int:
    best = seed
    for v in struct.unpack_from(_ROW_FMT[fmt], snap_acc, 0)[:active]:
        if v > best:
            best = v
    return best


def execute_agg_sum_first(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.SUM.FIRST: sum active MULT_RES elements, write to R_ACC[dest] (clean init)."""
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    fmt = lane_format(ipu.state)
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    active = agg_lane_count(valid_elements)
    result = _agg_sum_lanes(fmt, mult_res, active)
    dest = int(dest_slot) % LANES
    if fmt == "<i":
        result = _to_int32(result)
    struct.pack_into(fmt, ipu.state.regfile.mutable("r_acc", dest * 4, 4), 0, result)


def execute_agg_sum(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.SUM: sum active MULT_RES elements and add to R_ACC[dest] (running accumulation)."""
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    fmt = lane_format(ipu.state)
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    active = agg_lane_count(valid_elements)
    dest = int(dest_slot) % LANES
    snap_dest = struct.unpack_from(fmt, ipu._view.read_forgetting_provenance("r_acc"), dest * 4)[0]
    partial = _agg_sum_lanes(fmt, mult_res, active)
    if fmt == "<f":
        result: float | int = float(partial) + float(snap_dest)
    else:
        # fmt == "<i" covers both narrow INT8 mode and wide-vector INT32 mode
        # (lane_format); ipu_add's DType.INT8 branch is plain 32-bit wrap
        # with no 8-bit saturation, so it's correct for INT32 lanes too.
        result = ipu_add(_to_int32(partial), int(snap_dest), DType.INT8)
    struct.pack_into(fmt, ipu.state.regfile.mutable("r_acc", dest * 4, 4), 0, result)


def execute_agg_max_first(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.MAX.FIRST: max of active MULT_RES elements, write to R_ACC[dest] (no seed).

    When no elements are active (valid_elements=0) the identity seed
    (INT32_MIN / -inf) is written, so the destination is always defined.
    """
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    fmt = lane_format(ipu.state)
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    active = agg_lane_count(valid_elements)
    seed: float | int = -2147483648 if fmt == "<i" else float("-inf")
    result = _agg_max_lanes(fmt, mult_res, active, seed)
    dest = int(dest_slot) % LANES
    struct.pack_into(fmt, ipu.state.regfile.mutable("r_acc", dest * 4, 4), 0, result)


def execute_agg_max(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.MAX: max of active MULT_RES elements seeded with R_ACC[dest] (running max)."""
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    fmt = lane_format(ipu.state)
    mult_res = ipu._view.read_forgetting_provenance("mult_res")
    active = agg_lane_count(valid_elements)
    dest = int(dest_slot) % LANES
    snap_dest = struct.unpack_from(fmt, ipu._view.read_forgetting_provenance("r_acc"), dest * 4)[0]
    result = _agg_max_lanes(fmt, mult_res, active, snap_dest)
    struct.pack_into(fmt, ipu.state.regfile.mutable("r_acc", dest * 4, 4), 0, result)
