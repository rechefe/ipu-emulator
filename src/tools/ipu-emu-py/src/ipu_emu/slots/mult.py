"""MULT slot handlers, the lane mask they apply, and their issue-time probes.

The handlers write MULT_RES and then apply the R_MASK lane mask
(:func:`_mult_mask_and_shift`). The probes (:func:`scalar_is_one`,
:func:`vector_is_one`, :func:`alias_masked_window`) answer the issuer's
identity and alias questions with these handlers' own operand resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import effective_mult_mask, vector_registers, xmem_row_size_bytes
from ipu_emu.isa_alias_spec import VECTOR_IDENTITY_MULTS
from ipu_isa.constants import DStructureConfig, PadMode
from ipu_isa.geometry import LANES
from ipu_isa.registers import get_mult_stage_map, get_register_sizes

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_emu.numerics import LaneCodec
    from ipu_emu.numerics.codecs import Lanes

    Scalar = np.generic | int | float

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

# MultStageRegField: index → (register_name, element_index)
_MULT_STAGE_MAP: list[tuple[str, int]] = get_mult_stage_map()

# How a mult instruction's scalar multiplicand is addressed, if it has one
# (``SlotPlan.scalar_kind``). Which operand carries it comes from
# SCALAR_MULT_OPERANDS; how it is addressed is read off that operand's
# declared type, so neither fact is restated here.
SCALAR_NONE = 0
SCALAR_LCR = 1  # LcrIdx: an LR selects an Ra element, or a CR supplies the byte
SCALAR_CR = 2  # CrIdx: the CR's low byte is the scalar

# 0..LANES-1, for the rotated Ra and R_CYCLIC windows.
_LANE_INDEX = np.arange(LANES)

# Ra (R0 ++ R1) holds two rows of lanes. The R_CYCLIC ring holds 512 elements
# in every element width: ``r_cyclic`` is its 1-byte-element form.
_RA_LANES = 2 * LANES
_RING_LANES = _reg_sizes["r_cyclic"]["size_bytes"]

# One R_MASK slot (128 bits) with every lane active.
_ALL_ONES_MASK_SLOT = b"\xff" * 16
_ALL_ONES_MASK_INT = (1 << LANES) - 1


@lru_cache(maxsize=512)
def _inactive_lanes(mask_int: int) -> np.ndarray:
    """Lanes a 128-bit multiply mask deactivates, in ascending order (read-only).

    A mask that activates every lane (the default R_MASK, and what every
    kernel in the tree runs with) yields an empty array, so the caller writes
    no pad values.
    """
    lanes = np.array([i for i in range(LANES) if not ((mask_int >> i) & 1)], dtype=np.intp)
    lanes.flags.writeable = False
    return lanes


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
# Operand lanes, decoded through the active codec
# ---------------------------------------------------------------------------


def _ra_lanes(ipu: Ipu, codec: LaneCodec) -> Lanes:
    """Ra (R0 ++ R1) as one 256-lane array in the compute domain, from its declared view."""
    return codec.decode(ipu._view.read(vector_registers(ipu.state).ra), _RA_LANES)


def _r_cyclic_window(ipu: Ipu, codec: LaneCodec, rc_idx: int) -> Lanes:
    """LANES elements of R_CYCLIC from element ``rc_idx``, wrapping at the ring's end.

    ``rc_idx`` (the MULT.RC.* operand) addresses R_CYCLIC in elements, as
    ``LD.MULT.CYCLIC``'s ``index`` operand does, so reads and writes share one
    unit in every element width.
    """
    buf = ipu._view.read(vector_registers(ipu.state).r_cyclic)
    start = rc_idx % _RING_LANES
    if start + LANES <= _RING_LANES:
        return codec.decode(buf[start * codec.element_bytes :], LANES)
    return codec.decode(buf, _RING_LANES)[(start + _LANE_INDEX) % _RING_LANES]


def _store_mult_res(ipu: Ipu, codec: LaneCodec, product: Lanes) -> None:
    """Write LANES products to MULT_RES, rounded once to the accumulator type."""
    codec.store_acc(ipu.state.regfile.mutable("mult_res"), product)


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
    require a float accumulator — an integer one has no infinity.

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
        if ipu._view.read("r_mask")[slot : slot + 16] == _ALL_ONES_MASK_SLOT:
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
    if not inactive.size:
        return
    pad_word = np.frombuffer(_mult_pad_lane_bytes(ipu, dstructure.pad_mode), dtype="<u4")[0]
    np.frombuffer(ipu.state.regfile.mutable("mult_res"), dtype="<u4")[inactive] = pad_word


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

    ZERO is representable in every accumulator type. POS_INF/NEG_INF exist
    only in a float accumulator, so they are rejected under integer element
    arithmetic (INT8, and INT32 in wide-vector debug mode).
    """
    if pad_mode == PadMode.ZERO:
        return b"\x00\x00\x00\x00"
    acc_dtype = ipu._codec.acc_dtype
    if acc_dtype.kind != "f":
        raise EmulatorError(
            f"dstructure pad_mode {pad_mode.name} requires floating-point lanes; "
            "integer lane arithmetic has no infinity representation"
        )
    value = float("inf") if pad_mode == PadMode.POS_INF else float("-inf")
    return np.array(value, dtype=acc_dtype).tobytes()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def execute_mult_nop(ipu: Ipu) -> None:
    """Execute NOP in mult slot: No operation."""


def _mult_resolve_lcr_scalar(ipu: Ipu, codec: LaneCodec, src: int, via: str = "src") -> Scalar:
    """Resolve an LcrIdx ``src`` field that addresses an element scalar.

    If ``src`` encodes an LR, the LR's stored value is itself used as an
    index (mod 256) into the combined Ra buffer (``R0`` ++ ``R1``). If
    ``src`` encodes a CR, the CR word is the scalar, decoded by
    :meth:`LaneCodec.decode_scalar`. Each read has the view the instruction
    declares for it (the LR and the CR through the operand *via*, Ra as
    ``r``).
    """
    if src < LR_REG_COUNT:
        # The LR holds an INDEX into Ra; the Ra DATA is a separate read.
        return _ra_lanes(ipu, codec)[ipu._view.lr(src, via) % _RA_LANES]
    return codec.decode_scalar(ipu._view.cr(src - LR_REG_COUNT, via))


def execute_mult_rc_vv(
    ipu: Ipu, *, rc_idx: int, ra: bytearray, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VV: R_CYCLIC vector × Ra (R0/R1) vector, element-wise."""
    codec = ipu._codec
    rc = _r_cyclic_window(ipu, codec, rc_idx)
    _store_mult_res(ipu, codec, codec.mult(rc, codec.decode(ra)))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_rc_ve(
    ipu: Ipu, *, rc_idx: int, src: int, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VE: R_CYCLIC vector × scalar (R0/R1 element or CR value)."""
    codec = ipu._codec
    scalar = _mult_resolve_lcr_scalar(ipu, codec, src)
    rc = _r_cyclic_window(ipu, codec, rc_idx)
    _store_mult_res(ipu, codec, codec.mult(rc, scalar))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_rc_vs(
    ipu: Ipu, *, rc_idx: int, mask_offset: int, mask_shift: int, cr_idx: int
) -> None:
    """Execute MULT.RC.VS: R_CYCLIC vector self-multiply (square), element-wise."""
    codec = ipu._codec
    rc = _r_cyclic_window(ipu, codec, rc_idx)
    _store_mult_res(ipu, codec, codec.mult(rc, rc))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, cr_idx)


def execute_mult_ve(
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, dstructure_cr_idx: int
) -> None:
    """Execute MULT.VE: Ra (combined R0/R1) vector × CR scalar, element-wise."""
    codec = ipu._codec
    scalar = codec.decode_scalar(ipu._view.cr(cr_idx, "cr_idx"))
    # Ra is the R0 ++ R1 pair read as one 256-lane cyclic window.
    window = _ra_lanes(ipu, codec)[(ra_idx + _LANE_INDEX) % _RA_LANES]
    _store_mult_res(ipu, codec, codec.mult(window, scalar))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, dstructure_cr_idx)


def execute_mult_ee(
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, dstructure_cr_idx: int
) -> None:
    """Execute MULT.EE: single Ra element × CR scalar, broadcast to all 128 elements."""
    codec = ipu._codec
    scalar = codec.decode_scalar(ipu._view.cr(cr_idx, "cr_idx"))
    pos = ra_idx % _RA_LANES
    # One product, broadcast to every lane.
    product = codec.mult(_ra_lanes(ipu, codec)[pos : pos + 1], scalar)
    _store_mult_res(ipu, codec, np.repeat(product, LANES))
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
    codec = ipu._codec
    registers = vector_registers(ipu.state)
    size = xmem_row_size_bytes(ipu.state)
    ring = registers.r_cyclic
    start = kwargs["rc_idx"] * codec.element_bytes
    source = ipu._view.source(ring)
    if source.has_constant_ones(ring, start, size) and _is_ones_row(
        codec, _cyclic_bytes(source.view(ring), start, size)
    ):
        return True
    if name == "MULT.RC.VV":
        # ra is already resolved to bytes; the issuer kept its index.
        index = _MULT_STAGE_MAP[encoded["ra"]][1]
        reg = registers.ra
        snap = ipu._view.source(reg, "ra")
        if snap.has_constant_ones(reg, index * size, size):
            return _is_ones_row(codec, snap.get_register_bytes(reg, index))
    return False


def _cyclic_bytes(buf: memoryview, start: int, size: int) -> bytes:
    """``size`` bytes of a ring from byte ``start``, wrapping at its end."""
    start %= len(buf)
    head = bytes(buf[start : start + size])
    return head + bytes(buf[: size - len(head)])


def _is_ones_row(codec: LaneCodec, data: bytes | bytearray) -> bool:
    return data == codec.one() * LANES


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

    Compares the decoded scalar with 1, so "one" is the active format's own
    encoding of 1.

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
    return bool(_mult_resolve_lcr_scalar(ipu, ipu._codec, lcr, operand) == 1)
