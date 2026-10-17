"""MULT slot handlers, the lane mask they apply, and their issue-time probes.

The handlers write MULT_RES and then apply the R_MASK lane mask
(:func:`_mult_mask_and_shift`). The probes (:func:`scalar_is_one`,
:func:`vector_is_one`, :func:`alias_masked_window`) answer the issuer's
identity and alias questions with these handlers' own operand resolution.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import (
    cr_scalar as _geometry_cr_scalar,
)
from ipu_emu.geometry import (
    effective_mult_mask,
    store_wide_lanes,
    wide_lanes,
    xmem_element_width_bytes,
    xmem_row_size_bytes,
)
from ipu_emu.ipu_math import DType, dtype_one_byte, ipu_mult
from ipu_emu.ipu_state import WideVectorArithmetic
from ipu_emu.isa_alias_spec import VECTOR_IDENTITY_MULTS
from ipu_isa.constants import DStructureConfig, PadMode
from ipu_isa.geometry import LANES
from ipu_isa.registers import get_mult_stage_map, get_register_sizes

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu

__all__ = [
    "SCALAR_CR",
    "SCALAR_LCR",
    "SCALAR_NONE",
    "alias_masked_window",
    "scalar_is_one",
    "vector_is_one",
]

_reg_sizes = get_register_sizes()
LR_REG_COUNT = _reg_sizes["lr"]["count"]
R_REG_SIZE = _reg_sizes["r"]["size_bytes"]

# MultStageRegField: index → (register_name, element_index)
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()

# How a mult instruction's scalar multiplicand is addressed, if it has one
# (``SlotPlan.scalar_kind``). Which operand carries it comes from
# SCALAR_MULT_OPERANDS; how it is addressed is read off that operand's
# declared type, so neither fact is restated here.
SCALAR_NONE = 0
SCALAR_LCR = 1  # LcrIdx: an LR selects an Ra element, or a CR supplies the byte
SCALAR_CR = 2  # CrIdx: the CR's low byte is the scalar

# 0..LANES-1, for the rotated Ra window MULT.VE reads.
_LANE_INDEX = np.arange(LANES)

# One R_MASK slot (128 bits) with every lane active.
_ALL_ONES_MASK_SLOT = b"\xff" * 16
_ALL_ONES_MASK_INT = (1 << LANES) - 1


@lru_cache(maxsize=512)
def _inactive_lanes(mask_int: int) -> tuple[int, ...]:
    """Lanes a 128-bit multiply mask deactivates, in ascending order.

    A mask that activates every lane (the default R_MASK, and what every
    kernel in the tree runs with) yields an empty tuple, so the caller writes
    no pad values -- exactly what the per-lane scan did, without the scan.
    """
    return tuple(i for i in range(LANES) if not ((mask_int >> i) & 1))


@lru_cache(maxsize=8)
def _one_byte_for(dtype: DType) -> int:
    """Raw encoding of 1.0 for a dtype. Cached: the FP8 path re-encodes."""
    return dtype_one_byte(dtype)


@lru_cache(maxsize=1024)
def _active_lane_count(mask_int: int, valid_elements: int) -> int:
    """Lanes the multiply retires, capped at the declared ``valid_elements``.

    ``valid_elements`` does not gate the multiply itself — only AGG ops consult
    it — but a kernel that declares a narrower row is telling us how many lanes
    carry real data, so statistics honour it as a *width*, not as a position.

    The mask decides which lanes those are, and a scatter puts them wherever
    ``mask_offset`` points: writing partition p lands on lanes p*ps.., outside
    0..n-1. Intersecting with lanes 0..n-1 (``agg_lane_count``'s
    contiguous-from-zero rule, which is right for a reduction) would score every
    partition but the first as zero, so the count is the mask's own population
    instead. Where the mask already lies inside 0..n-1 the two agree.

    A CR carrying no dstructure configuration decodes to ``valid_elements == 0``,
    which is not a declaration of "no data" — a multiply naming such a CR for its
    partition/pad settings still retires every masked-in lane. Fall back to the
    mask there, so an undeclared register cannot silently zero the metric.
    """
    active = mask_int.bit_count()
    n = min(int(valid_elements), LANES)
    return active if n <= 0 else min(active, n)


# ---------------------------------------------------------------------------
# Wide-vector debug mode (emulator-only; GitHub issue #33)
# ---------------------------------------------------------------------------


def _wide_assert_lane_aligned_byte_offset(name: str, byte_off: int) -> None:
    """Wide-vector mode treats r_cyclic in 4-byte elements; misaligned offsets corrupt unpacking."""
    if byte_off % 4 != 0:
        raise EmulatorError(f"Wide-vector debug: {name} must be 4-byte aligned, got {byte_off}")


def _wide_imult32(a: int, b: int) -> int:
    p = int(a) * int(b)
    p &= 0xFFFFFFFF
    return p - 0x100000000 if p >= 0x80000000 else p


def _wide_ra_lanes(ipu: Ipu, mult_stage_enc: int) -> np.ndarray:
    """R0/R1 wide lanes from the cycle-start snapshot (issue #157).

    Read an immutable copy of snapshot storage: R0 and R1 are the two
    512-byte elements of ``r_wide_debug``, and ``wide_lanes`` makes
    its own widened copy, so nothing aliases the register.
    """
    return wide_lanes(ipu.state, ipu._view.read("r_wide_debug"), mult_stage_enc * LANES * 4)


def _wide_ra_lane(ipu: Ipu, idx: int) -> float | int:
    """One lane of the R0 ++ R1 pair (``0 <= idx < 2*LANES``) from the snapshot.

    The pair is contiguous in ``r_wide_debug`` (R0 first), so lane *idx*
    is the 4-byte element at ``4 * idx`` whichever register it falls in.
    """
    fmt = "<f" if ipu.state.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<i"
    return struct.unpack_from(fmt, ipu._view.read("r_wide_debug"), 4 * idx)[0]


def _wide_rb_lanes(ipu: Ipu, cyclic_byte_off: int) -> np.ndarray:
    """r_cyclic wide lanes from the cycle-start snapshot."""
    _wide_assert_lane_aligned_byte_offset("cyclic_offset", cyclic_byte_off)
    return wide_lanes(
        ipu.state,
        ipu._view.source("r_cyclic_wide_debug").get_r_cyclic_wide_debug_at(
            cyclic_byte_off, xmem_row_size_bytes(ipu.state)
        ),
    )


def _rc_element_to_byte_offset(ipu: Ipu, rc_idx: int) -> int:
    """Scale an r_cyclic ELEMENT index to a byte offset (mode-dependent width).

    ``rc_idx`` (MULT.RC.* operand) addresses r_cyclic the same way
    ``LD.MULT.CYCLIC``'s ``index`` operand does -- in elements, not
    bytes -- so reads and writes to the same register share one unit. A
    no-op in narrow mode (1 byte/element); scales by 4 in wide-vector
    debug mode. See issue #182's follow-up: rc_idx used to be a raw byte
    offset, which silently diverged from the write-side element index
    once r_cyclic's wide-mode ring grew to 2048 B (512 elements x 4 B).
    """
    return rc_idx * xmem_element_width_bytes(ipu.state)


# ---------------------------------------------------------------------------
# The lane mask
# ---------------------------------------------------------------------------


def _mult_mask_and_shift(ipu: Ipu, mask_idx: int, shift: int, cr_idx: int) -> None:
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
    if ipu._prototyping and shift == 0:
        # Prototyping shortcut: an unshifted all-ones mask slot deactivates
        # no lane, so the whole mask pipeline below leaves MULT_RES exactly
        # as the multiply wrote it. The hardware still applies the mask.
        # Lane statistics are still recorded: prototyping mode is only
        # allowed to skip work, never to change what a run reports.
        slot = (mask_idx % (LANES // 16)) * 16
        if ipu._view.read_forgetting_provenance("r_mask")[slot : slot + 16] == _ALL_ONES_MASK_SLOT:
            _count_mult_lanes(ipu, _ALL_ONES_MASK_INT, ipu._view.dstructure(cr_idx))
            return

    facts = ipu.issuer.current
    if facts is not None and facts.mask is not None:
        # The mask the issue step read for this op (IssueFacts.mask).
        mask_int, dstructure = facts.mask, ipu._view.dstructure(cr_idx)
    else:
        mask_int, dstructure = _effective_mult_mask(ipu, mask_idx, shift, cr_idx)
    _count_mult_lanes(ipu, mask_int, dstructure)

    inactive = _inactive_lanes(mask_int)
    if not inactive:
        return
    pad_bytes = _mult_pad_lane_bytes(ipu, dstructure.pad_mode)
    mult_res = ipu.state.regfile.mutable("mult_res")
    for i in inactive:
        mult_res[i * 4 : i * 4 + 4] = pad_bytes


def _count_mult_lanes(ipu: Ipu, mask_int: int, dstructure: DStructureConfig) -> None:
    """Record the lanes a multiply retires under ``mask_int``.

    Every multiply uses this lane count, including declared identities.
    """
    stats = ipu.state.stats
    lanes = _active_lane_count(mask_int, dstructure.valid_elements)
    stats.mult_lane_ops += lanes
    facts = ipu.issuer.current
    if facts is not None and facts.mult_identity is not None and facts.mult_identity.any:
        stats.mult_identity_lane_ops += lanes


def _effective_mult_mask(
    ipu: Ipu, mask_idx: int, shift: int, cr_idx: int
) -> tuple[int, DStructureConfig]:
    """The mask :func:`~ipu_emu.geometry.effective_mult_mask` gives, and CR *cr_idx*'s dstructure."""
    dstructure = ipu._view.dstructure(cr_idx)
    return effective_mult_mask(ipu._view.source("r_mask"), dstructure, mask_idx, shift), dstructure


def _mult_pad_lane_bytes(ipu: Ipu, pad_mode: PadMode) -> bytes:
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
    if not _lanes_are_float(ipu):
        raise EmulatorError(
            f"dstructure pad_mode {pad_mode.name} requires floating-point lanes; "
            "integer lane arithmetic has no infinity representation"
        )
    value = float("inf") if pad_mode == PadMode.POS_INF else float("-inf")
    return struct.pack("<f", value)


def _lanes_are_float(ipu: Ipu) -> bool:
    """Whether MULT_RES elements hold floats, under whichever mode is active."""
    if ipu.state.wide_vector_debug:
        return ipu.state.wide_vector_arithmetic == WideVectorArithmetic.FP32
    return ipu.state.dtype != DType.INT8


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def execute_mult_nop(ipu: Ipu) -> None:
    """Execute NOP in mult slot: No operation."""


def _mult_resolve_lcr_scalar(ipu: Ipu, src: int, via: str = "src") -> int:
    """Resolve an LcrIdx ``src`` field that addresses a *byte* scalar.

    If ``src`` encodes an LR, the LR's stored value is itself used as an
    index (mod 256) into the combined Ra buffer (``R0`` ++ ``R1``). If
    ``src`` encodes a CR, the CR's low byte is the scalar directly. Each
    read has the view the instruction declares for it (the LR and the CR
    through the operand *via*, Ra as ``r``).
    """
    if src < LR_REG_COUNT:
        # The LR holds an INDEX into Ra; the Ra DATA is a separate read.
        idx = ipu._view.lr(src, via) % (2 * R_REG_SIZE)
        return ipu._view.read("r")[idx]
    return ipu._view.cr(src - LR_REG_COUNT, via) & 0xFF


def _mult_resolve_lcr_scalar_wide(ipu: Ipu, src: int, via: str = "src") -> float | int:
    """Wide-vector counterpart of ``_mult_resolve_lcr_scalar``."""
    if src < LR_REG_COUNT:
        return _wide_ra_lane(ipu, ipu._view.lr(src, via) % (2 * LANES))
    cr_idx = src - LR_REG_COUNT
    cr_scalar = _geometry_cr_scalar(ipu._view.source("cr", via), cr_idx)
    if ipu.state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
        return float(cr_scalar)
    return cr_scalar


def execute_mult_rc_vv(
    ipu: Ipu, *, rc_idx: int, ra: bytearray | int, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VV: R_CYCLIC vector × Ra (R0/R1) vector, element-wise."""
    state = ipu.state
    mult_res = state.regfile.mutable("mult_res")

    if state.wide_vector_debug:
        byte_off = _rc_element_to_byte_offset(ipu, rc_idx)
        rb_lanes = _wide_rb_lanes(ipu, byte_off)
        ra_lanes = _wide_ra_lanes(ipu, ra)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                product = rb_lanes * ra_lanes
        else:
            product = rb_lanes * ra_lanes
        store_wide_lanes(state, mult_res, product)
        _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)
        return

    dtype = state.dtype
    rc = ipu._view.source("r_cyclic").get_r_cyclic_at(
        _rc_element_to_byte_offset(ipu, rc_idx), R_REG_SIZE
    )

    for i in range(LANES):
        result = ipu_mult(rc[i], ra[i], dtype)
        struct.pack_into("<i" if dtype == DType.INT8 else "<f", mult_res, i * 4, result)

    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_rc_ve(
    ipu: Ipu, *, rc_idx: int, src: int, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VE: R_CYCLIC vector × scalar (R0/R1 element or CR value)."""
    state = ipu.state
    mult_res = state.regfile.mutable("mult_res")

    if state.wide_vector_debug:
        byte_off = _rc_element_to_byte_offset(ipu, rc_idx)
        scalar = _mult_resolve_lcr_scalar_wide(ipu, src)
        rb_lanes = _wide_rb_lanes(ipu, byte_off)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                product = rb_lanes * float(scalar)
        else:
            product = rb_lanes * int(scalar)
        store_wide_lanes(state, mult_res, product)
        _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)
        return

    dtype = state.dtype
    scalar_byte = _mult_resolve_lcr_scalar(ipu, src)
    rc = ipu._view.source("r_cyclic").get_r_cyclic_at(
        _rc_element_to_byte_offset(ipu, rc_idx), R_REG_SIZE
    )
    fmt = "<i" if dtype == DType.INT8 else "<f"

    for i in range(LANES):
        result = ipu_mult(rc[i], scalar_byte, dtype)
        struct.pack_into(fmt, mult_res, i * 4, result)

    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_rc_vs(
    ipu: Ipu, *, rc_idx: int, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VS: R_CYCLIC vector self-multiply (square), element-wise."""
    state = ipu.state
    mult_res = state.regfile.mutable("mult_res")

    if state.wide_vector_debug:
        byte_off = _rc_element_to_byte_offset(ipu, rc_idx)
        rb_lanes = _wide_rb_lanes(ipu, byte_off)
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                product = rb_lanes * rb_lanes
        else:
            product = rb_lanes * rb_lanes
        store_wide_lanes(state, mult_res, product)
        _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)
        return

    dtype = state.dtype
    rc = ipu._view.source("r_cyclic").get_r_cyclic_at(
        _rc_element_to_byte_offset(ipu, rc_idx), R_REG_SIZE
    )
    fmt = "<i" if dtype == DType.INT8 else "<f"

    for i in range(LANES):
        result = ipu_mult(rc[i], rc[i], dtype)
        struct.pack_into(fmt, mult_res, i * 4, result)

    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_ve(
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, dstructure_cr_idx: int
) -> None:
    """Execute MULT.VE: Ra (combined R0/R1) vector × CR scalar, element-wise."""
    state = ipu.state
    mult_res = state.regfile.mutable("mult_res")

    if state.wide_vector_debug:
        cr_scalar = _geometry_cr_scalar(ipu._view.source("cr", "cr_idx"), cr_idx)
        # Ra is the R0 ++ R1 pair read as one 256-lane cyclic window.
        combined = np.concatenate((_wide_ra_lanes(ipu, 0), _wide_ra_lanes(ipu, 1)))
        window = combined[(ra_idx + _LANE_INDEX) % (2 * LANES)]
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            with np.errstate(all="ignore"):
                product = window * float(cr_scalar)
        else:
            product = window * int(cr_scalar)
        store_wide_lanes(state, mult_res, product)
        _mult_mask_and_shift(ipu, mask_offset, mask_shift, dstructure_cr_idx)
        return

    dtype = state.dtype
    scalar_byte = ipu._view.cr(cr_idx, "cr_idx") & 0xFF
    r_buf = ipu._view.read("r")  # Ra: [0:128]=r0, [128:256]=r1
    fmt = "<i" if dtype == DType.INT8 else "<f"

    for i in range(LANES):
        pos = (ra_idx + i) % (2 * R_REG_SIZE)
        result = ipu_mult(r_buf[pos], scalar_byte, dtype)
        struct.pack_into(fmt, mult_res, i * 4, result)

    _mult_mask_and_shift(ipu, mask_offset, mask_shift, dstructure_cr_idx)


def execute_mult_ee(
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, dstructure_cr_idx: int
) -> None:
    """Execute MULT.EE: single Ra element × CR scalar, broadcast to all 128 elements."""
    state = ipu.state
    mult_res = state.regfile.mutable("mult_res")

    if state.wide_vector_debug:
        cr_scalar = _geometry_cr_scalar(ipu._view.source("cr", "cr_idx"), cr_idx)
        ra_lane = _wide_ra_lane(ipu, ra_idx % (2 * LANES))
        # The product is one scalar operation, broadcast to every lane.
        if state.wide_vector_arithmetic == WideVectorArithmetic.FP32:
            result = float(ra_lane) * float(cr_scalar)
        else:
            result = _wide_imult32(int(ra_lane), cr_scalar)
        store_wide_lanes(state, mult_res, np.full(LANES, result))
        _mult_mask_and_shift(ipu, mask_offset, mask_shift, dstructure_cr_idx)
        return

    dtype = state.dtype
    scalar_byte = ipu._view.cr(cr_idx, "cr_idx") & 0xFF
    r_buf = ipu._view.read("r")  # Ra: [0:128]=r0, [128:256]=r1
    fmt = "<i" if dtype == DType.INT8 else "<f"

    ra_byte = r_buf[ra_idx % (2 * R_REG_SIZE)]
    result = ipu_mult(ra_byte, scalar_byte, dtype)
    for i in range(LANES):
        struct.pack_into(fmt, mult_res, i * 4, result)

    _mult_mask_and_shift(ipu, mask_offset, mask_shift, dstructure_cr_idx)


# ---------------------------------------------------------------------------
# Issue-time probes (called by ipu_emu.issue)
# ---------------------------------------------------------------------------


def vector_is_one(
    ipu: Ipu, name: str, kwargs: Mapping[str, Any], encoded: Mapping[str, int]
) -> bool:
    """Only explicitly declared ONES loads establish vector provenance.

    Args:
        ipu: The engine.
        name: The multiply's mnemonic.
        kwargs: Its resolved operands.
        encoded: Its raw operand fields (``ra`` is read as its encoded index).

    Returns:
        Whether the multiply's vector multiplicand is a declared all-ones load.
    """
    if name not in VECTOR_IDENTITY_MULTS:
        return False
    wide = ipu.state.wide_vector_debug
    size = xmem_row_size_bytes(ipu.state)
    ring = "r_cyclic_wide_debug" if wide else "r_cyclic"
    start = _rc_element_to_byte_offset(ipu, kwargs["rc_idx"])
    source = ipu._view.source(ring)
    if source.has_constant_ones(ring, start, size):
        data = (
            source.get_r_cyclic_wide_debug_at(start, size)
            if wide
            else source.get_r_cyclic_at(start, size)
        )
        if _ones_bytes(ipu, data):
            return True
    if name == "MULT.RC.VV":
        # Narrow ra is already resolved to bytes; the issuer kept its index.
        index = encoded["ra"]
        reg, index = ("r_wide_debug", index) if wide else _MULT_STAGE_MAP[index]
        snap = ipu._view.source(reg, "ra")
        if snap.has_constant_ones(reg, index * size, size):
            return _ones_bytes(ipu, snap.get_register_bytes(reg, index))
    return False


def _ones_bytes(ipu: Ipu, data: bytes | bytearray) -> bool:
    if ipu.state.wide_vector_debug:
        fmt = "<f" if ipu.state.wide_vector_arithmetic == WideVectorArithmetic.FP32 else "<i"
        return data == struct.pack(fmt, 1) * LANES
    return data == bytes([_one_byte_for(ipu.state.dtype)]) * LANES


def alias_masked_window(ipu: Ipu, kwargs: Mapping[str, Any]) -> bool:
    """Whether a MULT's lane mask selects a proper, non-empty part of its declared row.

    Args:
        ipu: The engine.
        kwargs: The multiply's resolved operands.

    Returns:
        ``True`` for a nonzero ``mask_offset``, or a mask that keeps some but
        not all of the ``valid_elements`` lanes.
    """
    if kwargs["mask_offset"] != 0:
        return True
    descriptor = kwargs.get("dstructure_cr_idx", kwargs.get("cr_idx"))
    mask, dstructure = _effective_mult_mask(ipu, 0, kwargs["mask_shift"], descriptor)
    valid = dstructure.valid_elements or LANES
    span = (1 << valid) - 1
    # Compare to the declared destination span, not the padded physical row.
    return bool(mask & span) and (mask & span) != span


def scalar_is_one(ipu: Ipu, kind: int, raw: int, operand: str) -> bool:
    """Whether a mult's multiplicand is a constant 1.0 taken from a CR.

    Only a *constant register* counts. An ``LcrIdx`` naming an LR selects an
    element of R0/R1, which is data — a quantized weight that happens to be
    1 is a real MAC, not a pass-through, and must not be written off as
    routing.

    Uses the dtype's own encoding of 1.0 because "one" is a different byte
    in every dtype, and a float under wide-vector FP32.

    Args:
        ipu: The engine.
        kind: How the multiplicand is addressed (``SlotPlan.scalar_kind``).
        raw: The operand's encoded value.
        operand: The operand's name (``SlotPlan.scalar_operand``), whose
            declared view the CR read takes.

    Returns:
        Whether the multiplicand is a CR holding 1.0.
    """
    # Normalise to an LcrIdx so the handlers' own resolvers decide the
    # value: they are the definition of what the multiplier will see.
    lcr = raw if kind == SCALAR_LCR else raw + LR_REG_COUNT
    if lcr < LR_REG_COUNT:
        return False
    if ipu.state.wide_vector_debug:
        return _mult_resolve_lcr_scalar_wide(ipu, lcr, operand) == 1
    return _mult_resolve_lcr_scalar(ipu, lcr, operand) == _one_byte_for(ipu.state.dtype)
