"""ACC slot handlers: accumulate (ACC.*), reduce (AGG.*), ACC.STRIDE and ACC.RESHAPE.

Every handler is one path for every element format (Q43, Q48): it loads
MULT_RES and R_ACC lanes through the active codec (``ipu._codec``), computes
with ``codec.add``/``sub``/``max`` and stores with ``codec.store_acc``, which
rounds once and fails lane by lane exactly as ``struct.pack_into`` did.
"""

from __future__ import annotations

import struct
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import agg_lane_count, lrd_bytes
from ipu_isa.acc_stride_enums import (
    get_elements_per_row,
    get_horizontal_stride_bits,
    get_vertical_stride_bits,
)
from ipu_isa.geometry import LANES
from ipu_isa.reshape_mask import RESHAPE_ELEMENT_COUNT, RESHAPE_MASK_LR_OFFSET

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_emu.numerics import LaneCodec

#: Bytes of one accumulator lane (int32 or float32) in every format.
_ACC_LANE_BYTES = 4


# ---------------------------------------------------------------------------
# ACC.*: accumulate MULT_RES into R_ACC
# ---------------------------------------------------------------------------


def execute_acc_nop(ipu: Ipu) -> None:
    """Execute NOP in acc slot: No operation."""


def _operands(ipu: Ipu) -> tuple[LaneCodec, np.ndarray, np.ndarray]:
    """The codec, R_ACC (this cycle's snapshot) and MULT_RES, as accumulator lanes."""
    codec = ipu._codec
    acc = codec.load_acc(ipu._view.read("r_acc"))
    res = codec.load_acc(ipu._view.read("mult_res"))
    return codec, acc, res


def _store_acc(ipu: Ipu, codec: LaneCodec, lanes: np.ndarray) -> None:
    codec.store_acc(ipu.state.regfile.mutable("r_acc"), lanes)


def _store_mult_res_in_acc(ipu: Ipu) -> None:
    """R_ACC = MULT_RES, through the accumulator load and store (a NaN is quietened)."""
    codec = ipu._codec
    res = codec.load_acc(ipu._view.read("mult_res"))
    _store_acc(ipu, codec, res)


def execute_acc_add(ipu: Ipu) -> None:
    """Execute ACC.ADD: Accumulate mult_res into accumulator (running add)."""
    codec, acc, res = _operands(ipu)
    _store_acc(ipu, codec, codec.add(acc, res))


def execute_acc_add_first(ipu: Ipu) -> None:
    """Execute ACC.ADD.FIRST: Set r_acc to multiply result (no previous sum)."""
    _store_mult_res_in_acc(ipu)


def execute_acc_max(ipu: Ipu) -> None:
    """Execute ACC.MAX: Each R_ACC element takes max(R_ACC[i], MULT_RES[i])."""
    codec, acc, res = _operands(ipu)
    _store_acc(ipu, codec, codec.max(acc, res))


def execute_acc_max_first(ipu: Ipu) -> None:
    """Execute ACC.MAX.FIRST: Overwrite each R_ACC element with MULT_RES (clean init for max)."""
    _store_mult_res_in_acc(ipu)


def execute_acc_sub(ipu: Ipu) -> None:
    """Execute ACC.SUB: Subtract MULT_RES from each R_ACC element (running subtract)."""
    codec, acc, res = _operands(ipu)
    _store_acc(ipu, codec, codec.sub(acc, res))


def execute_acc_sub_first(ipu: Ipu) -> None:
    """Execute ACC.SUB.FIRST: Set each R_ACC element to negated MULT_RES (clean init for subtract)."""
    codec = ipu._codec
    res = codec.load_acc(ipu._view.read("mult_res"))
    # 0 - x, not -x: for x = +0.0 that is +0.0, where negation flips the sign bit.
    _store_acc(ipu, codec, codec.sub(np.zeros_like(res), res))


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


@cache
def _stride_gather(
    elements_in_row: int, horizontal_stride: int, vertical_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """``_stride_indices`` as a gather: (MULT_RES lane per output lane, pad mask)."""
    indices = np.array(
        _stride_indices(elements_in_row, horizontal_stride, vertical_stride), dtype=np.intp
    )
    pad = indices < 0
    indices[pad] = 0
    indices.flags.writeable = False
    pad.flags.writeable = False
    return indices, pad


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
    A result that runs past R_ACC's last lane writes the lanes that fit, then
    raises ``struct.error`` (what the lane-by-lane ``struct.pack_into`` did).
    """
    indices, pad = _stride_gather(elements_in_row, horizontal_stride, vertical_stride)
    base = (offset % 4) * 32
    codec = ipu._codec
    res = codec.load_acc(ipu._view.read("mult_res"))
    values = np.where(pad, np.zeros_like(res[:1]), res[indices])
    # From the first lane written to the end of R_ACC.
    acc_buf = ipu.state.regfile.mutable("r_acc", base * _ACC_LANE_BYTES)
    fit = LANES - base
    codec.store_acc(acc_buf, values[:fit])
    if len(values) > fit:
        # The next lane does not fit: raise struct's own error, as the lane loop did.
        struct.pack_into("<I", acc_buf, fit * _ACC_LANE_BYTES, 0)


def execute_acc_reshape(ipu: Ipu, *, source: int, dest: int, reshape_mask: int) -> None:
    """Execute ACC.RESHAPE: scatter MULT_RES elements into R_ACC via two LRDn byte-index arrays.

    ``source``/``dest`` are LrdIdx pair indices (0-7); each resolves to 8 byte
    elements read after the same word's LR writes (forwarded, Q32). Only the trailing
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
        raise EmulatorError(
            f"ACC.RESHAPE reshape_mask value {mask} exceeds RESHAPE_ELEMENT_COUNT "
            f"({RESHAPE_ELEMENT_COUNT})"
        )

    mult_res = ipu._view.read("mult_res")
    writes = []
    for i in range(mask, RESHAPE_ELEMENT_COUNT):
        if src_bytes[i] >= LANES or dst_bytes[i] >= LANES:
            raise EmulatorError(
                f"ACC.RESHAPE element {i}: source={src_bytes[i]}, dest={dst_bytes[i]} "
                f"must both be in [0, {LANES - 1}]"
            )
        writes.append((dst_bytes[i], struct.unpack_from("<I", mult_res, src_bytes[i] * 4)[0]))
    for dest_idx, value in writes:
        ipu.state.regfile.set_r_acc_word(dest_idx, value)


# ---------------------------------------------------------------------------
# AGG.*: reduce MULT_RES into one R_ACC element
# ---------------------------------------------------------------------------


def _agg_inputs(ipu: Ipu, cr_idx: int, dest_slot: int) -> tuple[LaneCodec, list, int]:
    """The codec, the active MULT_RES lanes (Python numbers) and the R_ACC lane written."""
    valid_elements = ipu._view.dstructure(cr_idx, "cr_idx").valid_elements
    codec = ipu._codec
    active = agg_lane_count(valid_elements)
    lanes = codec.load_acc(ipu._view.read("mult_res"), active).tolist()
    return codec, lanes, int(dest_slot) % LANES


def _agg_dest(ipu: Ipu, codec: LaneCodec, dest: int) -> np.ndarray:
    """R_ACC[dest] from this cycle's snapshot, as one accumulator lane."""
    snap = ipu._view.read("r_acc")
    return codec.load_acc(snap[dest * _ACC_LANE_BYTES :], 1)


def _store_dest(ipu: Ipu, codec: LaneCodec, dest: int, value: np.ndarray | float | int) -> None:
    """Write one accumulator lane to R_ACC[dest]."""
    out = ipu.state.regfile.mutable("r_acc", dest * _ACC_LANE_BYTES, _ACC_LANE_BYTES)
    codec.store_acc(out, np.asarray(value).reshape(1))


def _fold_sum(lanes: list) -> float | int:
    # Summed in lane order, one lane at a time, in Python numbers: floating-
    # point addition is not associative, so a pairwise (numpy) sum would give
    # a different answer. Integer sums are exact; the store wraps them to int32.
    # (An int 0 start adds exactly as 0.0 to a float lane.)
    total: float | int = 0
    for value in lanes:
        total += value
    return total


def _fold_max(lanes: list, seed: float | int) -> float | int:
    # Python's max fold: a lane replaces the best only when strictly greater,
    # so a NaN seed stays and the first of equal values (+0.0, -0.0) wins.
    best = seed
    for value in lanes:
        if value > best:
            best = value
    return best


def execute_agg_sum_first(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.SUM.FIRST: sum active MULT_RES elements, write to R_ACC[dest] (clean init)."""
    codec, lanes, dest = _agg_inputs(ipu, cr_idx, dest_slot)
    _store_dest(ipu, codec, dest, _fold_sum(lanes))


def execute_agg_sum(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.SUM: sum active MULT_RES elements and add to R_ACC[dest] (running accumulation)."""
    codec, lanes, dest = _agg_inputs(ipu, cr_idx, dest_slot)
    partial = np.asarray([_fold_sum(lanes)])
    _store_dest(ipu, codec, dest, codec.add(partial, _agg_dest(ipu, codec, dest)))


def execute_agg_max_first(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.MAX.FIRST: max of active MULT_RES elements, write to R_ACC[dest] (no seed).

    When no elements are active (valid_elements=0) the identity seed
    (``codec.acc_min()``: INT32_MIN or -inf) is written, so the destination is always defined.
    """
    codec, lanes, dest = _agg_inputs(ipu, cr_idx, dest_slot)
    _store_dest(ipu, codec, dest, _fold_max(lanes, codec.acc_min().item()))


def execute_agg_max(ipu: Ipu, *, dest_slot: int, cr_idx: int) -> None:
    """Execute AGG.MAX: max of active MULT_RES elements seeded with R_ACC[dest] (running max)."""
    codec, lanes, dest = _agg_inputs(ipu, cr_idx, dest_slot)
    seed = _agg_dest(ipu, codec, dest)[0].item()
    _store_dest(ipu, codec, dest, _fold_max(lanes, seed))
