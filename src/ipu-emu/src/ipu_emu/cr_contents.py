"""What the datapath reads from a CR: DS and QF halves and MULT constants (ADR 0007).

The layouts are ISA data (``ipu_isa.cr_halves``); this module decodes them
for the handlers, caching each decode by value (a CR never changes during a
run, so a kernel decodes a handful of values millions of times), and turns an
invalid content into a :class:`~ipu_emu.errors.CrContentError`.

A half-CR operand (``ds``, ``qf``) is its ``HalfCrIdx`` code, ``2 * index +
high``. The handlers and the check predicates read the CR through the cycle
view (``CycleView.ds`` / ``CycleView.qf``), which call :func:`ds_of` and
:func:`qf_of` here.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

from ipu_emu.errors import CrContentError
from ipu_isa.constants import CR_ONE_REG_INDEX, CR_ZERO_REG_INDEX
from ipu_isa.cr_halves import (
    DataStructure,
    MaskType,
    decode_constant,
    decode_ds,
    decode_qf,
    half,
    half_cr_code,
)
from ipu_isa.errors import IsaValueError
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format, decode_element

__all__ = [
    "ZERO_FP32_ROW",
    "constant_value",
    "ds_active_count",
    "ds_active_mask",
    "ds_of",
    "float32_bytes",
    "fp32_constant_value",
    "half_of",
    "mask_pad_value",
    "qf_of",
]


def half_of(code: int, cr_word: int) -> int:
    """The 16-bit half a ``HalfCrIdx`` operand *code* selects from *cr_word*."""
    return half(cr_word, half_cr_code(code)[1])


@lru_cache(maxsize=256)
def ds_of(value: int) -> DataStructure:
    """The DS a 16-bit half holds.

    Raises:
        CrContentError: The half is no valid DS (``valid_partitions`` 0 or
            above ``num_partitions``, ``valid_elements`` above the group size,
            ``mask_type`` NaN).
    """
    try:
        return decode_ds(value)
    except IsaValueError as error:
        raise CrContentError(f"DS half {value:#06x} is invalid: {error}") from None


@lru_cache(maxsize=256)
def qf_of(value: int) -> Nx8Format:
    """The NX8 format a 16-bit QF half names.

    Raises:
        CrContentError: The half is no valid format (``num_bits`` 0, ``fe`` 0,
            ``fe + sign`` above ``num_bits``, a dynamic format not signed and 8
            bits wide, a ``dynamic`` field above 1).
    """
    try:
        return decode_qf(value)
    except IsaValueError as error:
        raise CrContentError(f"QF half {value:#06x} is invalid: {error}") from None


@lru_cache(maxsize=256)
def ds_active_mask(value: int) -> int:
    """The DS's active elements as a 128-bit mask (bit i: element i)."""
    return sum(1 << i for i, on in enumerate(ds_of(value).active()) if on)


@lru_cache(maxsize=256)
def ds_active_count(value: int) -> int:
    """How many of the 128 elements the DS makes active."""
    return ds_active_mask(value).bit_count()


#: The MULT_RES fill value of a masked element, per DS ``mask_type`` (NaN is refused).
_PAD = {MaskType.ZERO: 0.0, MaskType.POS_INF: float("inf"), MaskType.NEG_INF: float("-inf")}


def mask_pad_value(shape: DataStructure) -> float:
    """What MULT writes into a masked-out element under *shape*'s ``mask_type``."""
    return _PAD[shape.mask_type]


@lru_cache(maxsize=256)
def constant_value(cr_index: int, word: int) -> float:
    """CR *cr_index*, holding *word*, as an NX8-mode MULT constant.

    ``CR0`` reads 0.0 and ``CR1`` 1.0 (ADR 0007); any other CR is decoded with
    the constant layout: one 8-bit NX8 element with its own scale and type.

    Raises:
        CrContentError: The constant's type is invalid (``fe`` 0, or ``fe + sign``
            above 8).
    """
    if cr_index == CR_ZERO_REG_INDEX:
        return 0.0
    if cr_index == CR_ONE_REG_INDEX:
        return 1.0
    try:
        constant = decode_constant(word)
    except IsaValueError as error:
        raise CrContentError(f"CR{cr_index} holds no valid MULT constant: {error}") from None
    return decode_element(constant.value, constant.fmt, constant.scale)


def fp32_constant_value(cr_index: int, word: int) -> float:
    """CR *cr_index* as an FP32-mode MULT constant: 0.0, 1.0, else its float32 bits."""
    if cr_index == CR_ZERO_REG_INDEX:
        return 0.0
    if cr_index == CR_ONE_REG_INDEX:
        return 1.0
    with np.errstate(all="ignore"):
        return float(np.uint32(word & 0xFFFFFFFF).view("<f4"))


def float32_bytes(values: list[float]) -> bytes:
    """*values* as little-endian float32 bytes (an overflow becomes an infinity)."""
    with np.errstate(all="ignore"):
        return np.asarray(values, dtype=np.float64).astype("<f4").tobytes()


#: A row of LANES FP32 zeros: the inactive lanes' value in a stored FP32 row.
ZERO_FP32_ROW = struct.pack(f"<{LANES}f", *([0.0] * LANES))
