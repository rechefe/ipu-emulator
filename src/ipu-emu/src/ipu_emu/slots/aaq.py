"""AAQ slot handlers: ``ACTIVATE.QUANTIZE`` and ``LD.LUT`` (ADR 0007).

``ACTIVATE.QUANTIZE`` activates the forwarded R_ACC with the ARCH function
(``ipu_emu.activations.apply_aaq_activation``) and stores one row at
``CR[table_id] + LR[table_offset]``. In NX8 mode the active elements (the DS
half) are quantized to one NX8 row in the QF half's format by
``ipu_isa.nx8.encode_row`` (``ARCH-stage-aaq-str.md`` §4.3.2), and the row's
128 bytes and its metadata word are written; a saturation sets the
``QUANT_SATURATION`` arithmetic event. In FP32 mode the activated row is
stored as 128 FP32 values (512 bytes). Inactive elements are written 0 in both
modes (OQ-36).

``LD.LUT`` loads the first 128 bytes of a row into one 1024-bit segment of
the AaQ LUT (``aaq_lut``), in both modes (``ipu_isa.lut``).

The check predicates fault F-6 on a row outside XMEM and refuse an invalid DS
or QF (``CrContentError``) in the word's check phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ipu_emu.activations import apply_aaq_activation
from ipu_emu.cr_contents import float32_bytes
from ipu_emu.errors import EmulatorError
from ipu_emu.geometry import fp32_mode, xmem_row_addr, xmem_row_check
from ipu_isa.errors import IsaValueError
from ipu_isa.faults import ArithmeticEvent, FaultId
from ipu_isa.lut import SEGMENT_BYTES, TABLE_BITS, load_segment, segment_payload
from ipu_isa.nx8 import encode_row
from ipu_isa.registers import REGISTER_DEFINITIONS

if TYPE_CHECKING:
    from ipu_emu.ipu import Ipu
    from ipu_isa.cr_halves import DataStructure

Fault = tuple[FaultId, str]

#: Bytes of the AaQ LUT register: the table bits, then the LUT metadata.
_LUT_BYTES = REGISTER_DEFINITIONS["aaq_lut"]["size_bytes"]
_TABLE_MASK = (1 << TABLE_BITS) - 1
_SATURATION = frozenset({ArithmeticEvent.QUANT_SATURATION})


def execute_aaq_nop(ipu: Ipu) -> None:
    """Execute NOP in aaq slot: No operation."""


# ---------------------------------------------------------------------------
# LD.LUT
# ---------------------------------------------------------------------------


def check_ld_lut(ipu: Ipu, *, lut_addr: int, table_id: int, table_offset: int) -> Fault | None:
    """F-6 for a row outside XMEM (``lut_addr`` 5-7 is F-1 at fetch)."""
    return xmem_row_check(ipu.state, table_id + table_offset)


def execute_ld_lut(ipu: Ipu, *, lut_addr: int, table_id: int, table_offset: int) -> None:
    """Execute LD.LUT: one row's first 128 bytes into LUT segment ``lut_addr``."""
    state = ipu.state
    addr = xmem_row_addr(state, table_id + table_offset)
    payload = segment_payload(bytes(state.xmem.read_address(addr, SEGMENT_BYTES)))
    lut = state.regfile.mutable("aaq_lut")
    image = int.from_bytes(lut, "little")
    table, metadata = load_segment(image & _TABLE_MASK, lut_addr, payload)
    if metadata is None:
        metadata = image >> TABLE_BITS
    lut[:] = (table | (metadata << TABLE_BITS)).to_bytes(_LUT_BYTES, "little")


# ---------------------------------------------------------------------------
# ACTIVATE.QUANTIZE
# ---------------------------------------------------------------------------


def check_activate_quantize(
    ipu: Ipu, *, activation_fn: int, ds: int, qf: int, table_id: int, table_offset: int
) -> Fault | None:
    """The DS (and in NX8 the QF) must be valid; F-6 for a row outside XMEM."""
    ipu.view.ds(ds)
    if not fp32_mode(ipu.state):
        ipu.view.qf(qf)
    return xmem_row_check(ipu.state, table_id + table_offset)


def _activated(
    ipu: Ipu, activation_fn: int, shape: DataStructure
) -> tuple[list[float], list[bool]]:
    """The activated R_ACC row, rounded to FP32, and which elements are active.

    Inactive elements are 0.0 (they are written 0 and left out of the scale).
    """
    lanes = ipu.codec.load_acc(ipu.view.read("r_acc")).tolist()
    active = list(shape.active())
    options = ipu.state.aaq
    values = [
        apply_aaq_activation(activation_fn, x, options) if on else 0.0
        for x, on in zip(lanes, active, strict=True)
    ]
    with np.errstate(all="ignore"):  # an overflow is an infinity, as the FP32 datapath has
        rounded = np.asarray(values, dtype=np.float64).astype(np.float32).astype(np.float64)
    return rounded.tolist(), active


def execute_activate_quantize(
    ipu: Ipu, *, activation_fn: int, ds: int, qf: int, table_id: int, table_offset: int
) -> None:
    """Execute ACTIVATE.QUANTIZE: activate R_ACC, quantize (NX8) and store one row.

    Raises:
        EmulatorError: An activation raised (``exp2`` of a positive input,
            ``generic`` with no declared function), or NX8 mode and an
            activated element is NaN (NX8 has no NaN, OQ-37).
    """
    state = ipu.state
    values, active = _activated(ipu, activation_fn, ipu.view.ds(ds))
    addr = xmem_row_addr(state, table_id + table_offset)
    if fp32_mode(state):
        state.xmem.write_address(addr, float32_bytes(values))
        return
    try:
        row = encode_row(values, ipu.view.qf(qf), active)
    except IsaValueError as error:
        raise EmulatorError(f"ACTIVATE.QUANTIZE: {error}") from None
    state.xmem.write_address(addr, row.elements)
    state.xmem.write_metadata(addr, row.metadata)
    if row.saturated:
        state.record_arith(_SATURATION)
