"""MULT slot handlers, the lane mask they apply, and their issue-time probes.

The handlers write MULT_RES and then apply the R_MASK lane mask
(:func:`_mult_mask_and_shift`). In NX8 mode each vector element is expanded
to FP32 with the instruction's QF and the metadata of the row it came from
(``r_meta`` for R0/R1, ``r_cyclic_meta`` per R_CYCLIC slot; ADR 0007); a CR
scalar is a MULT constant (:func:`~ipu_emu.geometry.cr_scalar`). The DS half
gives the partitions of the mask shift and the pad of masked elements.

The check predicates (``check_<name>``) validate the DS, QF and constant in the
word's check phase, so an invalid one raises before anything of the word
retires. The probes (:func:`scalar_is_one`, :func:`vector_is_one`,
:func:`alias_masked_window`) answer the issuer's identity and alias questions
with these handlers' own operand resolution.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from ipu_emu.cr_contents import ds_active_count, ds_active_mask, mask_pad_value
from ipu_emu.geometry import (
    cr_scalar,
    effective_mult_mask,
    fp32_mode,
    vector_registers,
    xmem_row_size_bytes,
)
from ipu_emu.isa_alias_spec import VECTOR_IDENTITY_MULTS
from ipu_isa.geometry import LANES
from ipu_isa.registers import get_mult_stage_map, get_register_sizes

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_emu.numerics import LaneCodec
    from ipu_emu.numerics.codecs import Lanes
    from ipu_isa.cr_halves import DataStructure
    from ipu_isa.nx8 import Nx8Format

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
SCALAR_LCR = 1  # LcrIdx: an LR selects an Ra element, or a CR is the scalar
SCALAR_CR = 2  # CrIdx: the CR is the scalar (geometry.cr_scalar)

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
def _active_lane_count(mask_int: int, ds_active: int) -> int:
    """Lanes the multiply retires: the mask's lanes, capped at the DS's active count.

    The DS does not gate the multiply itself, but a kernel that declares a
    narrower row is telling us how many lanes carry real data, so statistics
    honour it as a *width*, not as a position: a scatter puts the masked lanes
    wherever ``mask_offset`` points, so the count is the mask's own population,
    capped at the DS's active elements.
    """
    return min(mask_int.bit_count(), ds_active)


# ---------------------------------------------------------------------------
# Operand lanes, decoded through the active codec
# ---------------------------------------------------------------------------


#: Bytes of one row-metadata word in ``r_meta`` / ``r_cyclic_meta``.
_META_BYTES = 2


def _qf(ipu: Ipu, qf: int) -> Nx8Format | None:
    """The NX8 format of the vector elements; ``None`` in FP32 mode (QF is ignored)."""
    if fp32_mode(ipu.state):
        return None
    return ipu.view.qf(qf)


def _metadata(ipu: Ipu, register: str, index: int) -> int:
    """Row-metadata word *index* of *register*, in its declared view."""
    raw = ipu.view.read(register)
    return int.from_bytes(raw[index * _META_BYTES : (index + 1) * _META_BYTES], "little")


def _decode_rows(
    ipu: Ipu, codec: LaneCodec, register: str, meta_register: str, rows: int, nx8: Nx8Format | None
) -> Lanes:
    """*rows* rows of LANES elements of *register*, each decoded with its own metadata."""
    buf = ipu.view.read(register)
    if nx8 is None:
        return codec.decode(buf, rows * LANES)
    step = LANES * codec.element_bytes
    return np.concatenate(
        [
            codec.decode(buf[k * step :], LANES, nx8, _metadata(ipu, meta_register, k))
            for k in range(rows)
        ]
    )


def _ra_lanes(ipu: Ipu, codec: LaneCodec, nx8: Nx8Format | None) -> Lanes:
    """Ra (R0 ++ R1) as one 256-lane array in the compute domain, from its declared view."""
    return _decode_rows(ipu, codec, vector_registers(ipu.state).ra, "r_meta", 2, nx8)


def _ra_row(ipu: Ipu, codec: LaneCodec, ra: bytearray, nx8: Nx8Format | None) -> Lanes:
    """The resolved ``ra`` operand (R0 or R1's bytes), decoded with that register's metadata."""
    if nx8 is None:
        return codec.decode(ra)
    index = _MULT_STAGE_MAP[ipu.issuer.current.encoded["ra"]][1]
    return codec.decode(ra, LANES, nx8, _metadata(ipu, "r_meta", index))


def _r_cyclic_window(ipu: Ipu, codec: LaneCodec, rc_idx: int, nx8: Nx8Format | None) -> Lanes:
    """LANES elements of R_CYCLIC from element ``rc_idx``, wrapping at the ring's end.

    ``rc_idx`` (the MULT.RC.* operand) addresses R_CYCLIC in elements, as
    ``LD.MULT.CYCLIC``'s ``index`` operand does, so reads and writes share one
    unit in every element width. In NX8 each element takes the metadata of its
    128-element slot.
    """
    start = rc_idx % _RING_LANES
    if nx8 is None and start + LANES <= _RING_LANES:
        buf = ipu.view.read(vector_registers(ipu.state).r_cyclic)
        return codec.decode(buf[start * codec.element_bytes :], LANES)
    ring = _decode_rows(
        ipu, codec, vector_registers(ipu.state).r_cyclic, "r_cyclic_meta", _RING_LANES // LANES, nx8
    )
    return ring[(start + _LANE_INDEX) % _RING_LANES]


def _store_mult_res(ipu: Ipu, codec: LaneCodec, product: Lanes) -> None:
    """Write LANES products to MULT_RES, rounded once to the accumulator type.

    The IEEE result is stored: a product beyond FP32's range is an infinity
    (NX8 elements reach that range through their row scale). MULT reports no
    arithmetic event; whether an overflow there is one is open (OQ-18).
    """
    codec.store_acc_events(ipu.state.regfile.mutable("mult_res"), product)


# ---------------------------------------------------------------------------
# The lane mask
# ---------------------------------------------------------------------------


def _mult_mask_and_shift(ipu: Ipu, mask_idx: int, shift: int, ds: int) -> None:
    """Apply sequential shift-and-AND mask generation, then gate mult_res.

    ``shift`` is interpreted as ``mask_shift_idx`` ∈ [−3, +3] (clamped).
    The slot mask M is taken from R_MASK slot ``mask_idx``, then shifted
    sequentially (one bit per step):
      idx 0  → M (unmodified)
      idx +k → shift left  k times, ANDing with partition_vector after each step
      idx −k → shift right k times, ANDing with inverse_partition_vector after each step

    Two partition vectors, both derived from the DS's ``num_partitions``:
      partition_vector         — 0 at the START of each group (used for left shifts)
      inverse_partition_vector — 0 at the END   of each group (used for right shifts)

    Elements in mult_res where the resulting mask bit is 0 are set to the
    DS's ``mask_type`` fill value (0, +inf or -inf).

    Mode-blind: the mask is 128 bits (one bit per element) and MULT_RES is
    128 FP32 elements in both modes, so the same code drives both.
    """
    view = ipu.view
    ds_value = view.half(ds, "ds")
    if ipu.prototyping and shift == 0:
        # Prototyping shortcut: an unshifted all-ones mask slot deactivates
        # no lane, so the whole mask pipeline below leaves MULT_RES exactly
        # as the multiply wrote it. The hardware still applies the mask.
        # Lane statistics are still recorded: prototyping mode is only
        # allowed to skip work, never to change what a run reports.
        slot = (mask_idx % (LANES // 16)) * 16
        if view.read("r_mask")[slot : slot + 16] == _ALL_ONES_MASK_SLOT:
            _count_mult_lanes(ipu, _ALL_ONES_MASK_INT, ds_value)
            return

    shape = view.ds(ds)
    facts = ipu.issuer.current
    if facts is not None and facts.mask is not None:
        mask_int = facts.mask  # the mask the issue step read for this op (IssueFacts.mask)
    else:
        mask_int = effective_mult_mask(view.source("r_mask"), shape, mask_idx, shift)
    _count_mult_lanes(ipu, mask_int, ds_value)

    inactive = _inactive_lanes(mask_int)
    if not inactive.size:
        return
    pad = np.array(mask_pad_value(shape), dtype="<f4").view("<u4")
    np.frombuffer(ipu.state.regfile.mutable("mult_res"), dtype="<u4")[inactive] = pad


def _count_mult_lanes(ipu: Ipu, mask_int: int, ds_value: int) -> None:
    """Record the lanes a multiply retires under ``mask_int`` and the DS half ``ds_value``.

    Every multiply uses this lane count, including declared identities.
    """
    stats = ipu.state.stats
    lanes = _active_lane_count(mask_int, ds_active_count(ds_value))
    stats.mult_lane_ops += lanes
    facts = ipu.issuer.current
    if facts is not None and facts.mult_identity is not None and facts.mult_identity.any:
        stats.mult_identity_lane_ops += lanes


def _effective_mult_mask(ipu: Ipu, mask_idx: int, shift: int, ds: int) -> tuple[int, DataStructure]:
    """The mask :func:`~ipu_emu.geometry.effective_mult_mask` gives, and the DS half *ds*."""
    shape = ipu.view.ds(ds)
    return effective_mult_mask(ipu.view.source("r_mask"), shape, mask_idx, shift), shape


# ---------------------------------------------------------------------------
# Check predicates: the DS, QF and constant must be valid (ADR 0007)
# ---------------------------------------------------------------------------


def _check_halves(ipu: Ipu, ds: int, qf: int) -> None:
    """Raise ``CrContentError`` unless the DS (and, in NX8, the QF) is valid."""
    ipu.view.ds(ds)
    _qf(ipu, qf)


def check_mult_rc_vv(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, rc_idx: int, ra: bytearray, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """MULT.RC.VV faults on nothing; its DS and QF must be valid."""
    _check_halves(ipu, ds, qf)


def check_mult_rc_ve(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, rc_idx: int, src: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """MULT.RC.VE faults on nothing; its DS, QF and a CR constant must be valid."""
    _check_halves(ipu, ds, qf)
    if src >= LR_REG_COUNT:
        cr_idx = src - LR_REG_COUNT
        cr_scalar(ipu.state, cr_idx, ipu.view.cr(cr_idx, "src"))


def check_mult_rc_vs(
    ipu: Ipu, *, rc_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """MULT.RC.VS faults on nothing; its DS and QF must be valid."""
    _check_halves(ipu, ds, qf)


def check_mult_ve(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """MULT.VE faults on nothing; its DS, QF and constant must be valid."""
    _check_halves(ipu, ds, qf)
    cr_scalar(ipu.state, cr_idx, ipu.view.cr(cr_idx, "cr_idx"))


def check_mult_ee(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """MULT.EE faults on nothing; its DS, QF and constant must be valid."""
    check_mult_ve(
        ipu,
        ra_idx=ra_idx,
        cr_idx=cr_idx,
        mask_offset=mask_offset,
        mask_shift=mask_shift,
        ds=ds,
        qf=qf,
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def execute_mult_nop(ipu: Ipu) -> None:
    """Execute NOP in mult slot: No operation."""


def _mult_resolve_lcr_scalar(
    ipu: Ipu, codec: LaneCodec, src: int, nx8: Nx8Format | None, via: str = "src"
) -> Scalar:
    """Resolve an LcrIdx ``src`` field that addresses an element scalar.

    If ``src`` encodes an LR, the LR's stored value is itself used as an
    index (mod 256) into the combined Ra buffer (``R0`` ++ ``R1``), decoded
    with ``nx8`` and its row's metadata. If ``src`` encodes a CR, the CR is a
    MULT constant (:func:`~ipu_emu.geometry.cr_scalar`). Each read has the view
    the instruction declares for it (the LR and the CR through the operand
    *via*, Ra as ``r``).
    """
    if src < LR_REG_COUNT:
        # The LR holds an INDEX into Ra; the Ra DATA is a separate read.
        return _ra_lanes(ipu, codec, nx8)[ipu.view.lr(src, via) % _RA_LANES]
    cr_idx = src - LR_REG_COUNT
    return cr_scalar(ipu.state, cr_idx, ipu.view.cr(cr_idx, via))


def execute_mult_rc_vv(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, rc_idx: int, ra: bytearray, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """Execute MULT.RC.VV: R_CYCLIC vector × Ra (R0/R1) vector, element-wise."""
    codec = ipu.codec
    nx8 = _qf(ipu, qf)
    rc = _r_cyclic_window(ipu, codec, rc_idx, nx8)
    _store_mult_res(ipu, codec, codec.mult(rc, _ra_row(ipu, codec, ra, nx8)))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, ds)


def execute_mult_rc_ve(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, rc_idx: int, src: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """Execute MULT.RC.VE: R_CYCLIC vector × scalar (R0/R1 element or CR constant)."""
    codec = ipu.codec
    nx8 = _qf(ipu, qf)
    scalar = _mult_resolve_lcr_scalar(ipu, codec, src, nx8)
    rc = _r_cyclic_window(ipu, codec, rc_idx, nx8)
    _store_mult_res(ipu, codec, codec.mult(rc, scalar))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, ds)


def execute_mult_rc_vs(
    ipu: Ipu, *, rc_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """Execute MULT.RC.VS: R_CYCLIC vector self-multiply (square), element-wise."""
    codec = ipu.codec
    rc = _r_cyclic_window(ipu, codec, rc_idx, _qf(ipu, qf))
    _store_mult_res(ipu, codec, codec.mult(rc, rc))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, ds)


def execute_mult_ve(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """Execute MULT.VE: Ra (combined R0/R1) vector × CR constant, element-wise."""
    codec = ipu.codec
    scalar = cr_scalar(ipu.state, cr_idx, ipu.view.cr(cr_idx, "cr_idx"))
    # Ra is the R0 ++ R1 pair read as one 256-lane cyclic window.
    window = _ra_lanes(ipu, codec, _qf(ipu, qf))[(ra_idx + _LANE_INDEX) % _RA_LANES]
    _store_mult_res(ipu, codec, codec.mult(window, scalar))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, ds)


def execute_mult_ee(  # noqa: PLR0913 -- one keyword per spec operand (instruction_spec.py)
    ipu: Ipu, *, ra_idx: int, cr_idx: int, mask_offset: int, mask_shift: int, ds: int, qf: int
) -> None:
    """Execute MULT.EE: single Ra element × CR constant, broadcast to all 128 elements."""
    codec = ipu.codec
    scalar = cr_scalar(ipu.state, cr_idx, ipu.view.cr(cr_idx, "cr_idx"))
    pos = ra_idx % _RA_LANES
    # One product, broadcast to every lane.
    product = codec.mult(_ra_lanes(ipu, codec, _qf(ipu, qf))[pos : pos + 1], scalar)
    _store_mult_res(ipu, codec, np.repeat(product, LANES))
    _mult_mask_and_shift(ipu, mask_offset, mask_shift, ds)


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
    codec = ipu.codec
    registers = vector_registers(ipu.state)
    size = xmem_row_size_bytes(ipu.state)
    ring = registers.r_cyclic
    start = kwargs["rc_idx"] * codec.element_bytes
    source = ipu.view.source(ring)
    if source.has_constant_ones(ring, start, size) and _is_ones_row(
        codec, _cyclic_bytes(source.view(ring), start, size)
    ):
        return True
    if name == "MULT.RC.VV":
        # ra is already resolved to bytes; the issuer kept its index.
        index = _MULT_STAGE_MAP[encoded["ra"]][1]
        reg = registers.ra
        snap = ipu.view.source(reg, "ra")
        if snap.has_constant_ones(reg, index * size, size):
            return _is_ones_row(codec, snap.get_register_bytes(reg, index))
    return False


def _cyclic_bytes(buf: memoryview, start: int, size: int) -> bytes:
    """``size`` bytes of a ring from byte ``start``, wrapping at its end."""
    start %= len(buf)
    head = bytes(buf[start : start + size])
    return head + bytes(buf[: size - len(head)])


#: 128 FP32 ones: an FP32 ONES row.
_FP32_ONES_ROW = np.ones(LANES, dtype="<f4").tobytes()


def _is_ones_row(codec: LaneCodec, data: bytes | bytearray) -> bool:
    """Whether a declared ONES row still encodes ones. An NX8 row's value
    depends on the QF and metadata it is read with, so its declaration
    (``IpuState.write_constant_ones``) is trusted."""
    if codec.element_bytes == 1:
        return True
    return bytes(data) == _FP32_ONES_ROW


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
    mask, _ = _effective_mult_mask(ipu, 0, kwargs["mask_shift"], kwargs["ds"])
    span = ds_active_mask(ipu.view.half(kwargs["ds"], "ds"))
    # Compare to the declared destination span (the DS's active elements),
    # not the padded physical row.
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
    return bool(_mult_resolve_lcr_scalar(ipu, ipu.codec, lcr, None, operand) == 1)
