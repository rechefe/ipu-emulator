"""CR contents the datapath decodes: QF and DS halves, and the MULT constant (ADR 0007).

A CR holds two 16-bit halves, ``LOW`` (bits 15:0) and ``HIGH`` (bits 31:16).
A QF half names an NX8 format and a DS half the shape of a 128-element vector;
an instruction operand says which half it reads (``CR13.L``, ``CR13.H``). The
layouts are the architects' provisional 16-bit drafts (OQ-40), authored once
here as data so the emulator, the assembler's docs and the harness agree.

The MULT constant is a full 32-bit CR: an NX8 element, its scale and its NX
type. ``CR0`` and ``CR1`` are not decoded with this layout: as MULT constants
they read 0.0 and 1.0 (the emulator applies that rule, as it knows the index).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Final

from ipu_isa.errors import IsaValueError
from ipu_isa.geometry import LANES
from ipu_isa.nx8 import Nx8Format

HALF_BITS: Final = 16
HALF_MASK: Final = (1 << HALF_BITS) - 1


@dataclass(frozen=True)
class Field:
    """One bit field: ``width`` bits starting at bit ``lsb``."""

    lsb: int
    width: int

    def get(self, word: int) -> int:
        return (word >> self.lsb) & ((1 << self.width) - 1)

    def put(self, value: int) -> int:
        if not 0 <= value < (1 << self.width):
            raise IsaValueError(f"{value} does not fit a {self.width}-bit field")
        return value << self.lsb


#: QF half: the NX8 format (ADR 0007).
QF_LAYOUT: Final[Mapping[str, Field]] = MappingProxyType(
    {
        "num_bits": Field(0, 4),
        "exp_format": Field(4, 4),
        "sign": Field(8, 4),
        "dynamic": Field(12, 4),
    }
)

#: DS half: the shape of a 128-element vector (ADR 0007).
DS_LAYOUT: Final[Mapping[str, Field]] = MappingProxyType(
    {
        "valid_elements": Field(0, 8),
        "num_partitions": Field(8, 2),  # 1, 2, 4, 8 encoded as 0-3
        "mask_type": Field(10, 2),
        "valid_partitions": Field(12, 4),
    }
)

#: MULT constant CR (32 bits): an NX8 element with its own scale and type.
CONSTANT_LAYOUT: Final[Mapping[str, Field]] = MappingProxyType(
    {
        "value": Field(0, 8),
        "scale": Field(8, 8),
        "exp_format": Field(16, 4),
        "sign": Field(20, 4),
    }
)


def half(word: int, high: bool) -> int:
    """The LOW or HIGH 16-bit half of a 32-bit CR value."""
    return (word >> HALF_BITS if high else word) & HALF_MASK


def sign_extend_half(value: int) -> int:
    """A 16-bit half as a signed 32-bit integer (LR and COND operands)."""
    value &= HALF_MASK
    return value - (1 << HALF_BITS) if value & 0x8000 else value


def pack_halves(low: int, high: int = 0) -> int:
    """A 32-bit CR value from two 16-bit halves."""
    for name, value in (("low", low), ("high", high)):
        if not 0 <= value <= HALF_MASK:
            raise IsaValueError(f"{name} half {value} is not 16 bits")
    return (high << HALF_BITS) | low


# --- QF ----------------------------------------------------------------------


def qf(fmt: Nx8Format) -> int:
    """The QF half of *fmt*."""
    f = QF_LAYOUT
    return (
        f["num_bits"].put(fmt.num_bits)
        | f["exp_format"].put(fmt.exp_format)
        | f["sign"].put(fmt.sign)
        | f["dynamic"].put(int(fmt.dynamic))
    )


def decode_qf(value: int) -> Nx8Format:
    """The NX8 format a QF half names; raises ``IsaValueError`` if it is invalid."""
    f = QF_LAYOUT
    dynamic = f["dynamic"].get(value)
    if dynamic > 1:
        raise IsaValueError(f"QF dynamic field {dynamic} is not 0 or 1")
    return Nx8Format(
        sign=f["sign"].get(value),
        exp_format=f["exp_format"].get(value),
        num_bits=f["num_bits"].get(value),
        dynamic=bool(dynamic),
    )


# --- DS ----------------------------------------------------------------------


class MaskType(IntEnum):
    """The value MULT writes into masked products (DS ``mask_type``)."""

    ZERO = 0
    POS_INF = 1
    NEG_INF = 2
    NAN = 3  # NX8 has no NaN (OQ-37): decoding it raises


#: ``num_partitions`` values, by their 2-bit code.
PARTITIONS: Final = (1, 2, 4, 8)


@dataclass(frozen=True)
class DataStructure:
    """A decoded DS half: groups, active elements and the mask pad value.

    ``valid_elements`` counts within each group; the first ``valid_partitions``
    groups are active. AGG reads ``valid_elements`` alone (the first
    ``valid_elements`` elements, ``ARCH-stage-acc.md`` §4.4).
    """

    valid_elements: int = LANES
    num_partitions: int = 1
    mask_type: MaskType = MaskType.ZERO
    valid_partitions: int = 1

    def __post_init__(self) -> None:
        reason = self.invalid_reason()
        if reason is not None:
            raise IsaValueError(f"invalid DS {self}: {reason}")

    def invalid_reason(self) -> str | None:
        """Why this DS is invalid, or ``None``."""
        if self.num_partitions not in PARTITIONS:
            return "num_partitions must be 1, 2, 4 or 8"
        if not 1 <= self.valid_partitions <= self.num_partitions:
            return "valid_partitions must be 1..num_partitions"
        if not 0 <= self.valid_elements <= self.group_size:
            return f"valid_elements must be 0..{self.group_size} (the group size)"
        if self.mask_type is MaskType.NAN:
            return "mask_type NaN: NX8 has no NaN (OQ-37)"
        return None

    @property
    def group_size(self) -> int:
        return LANES // self.num_partitions

    def active(self) -> tuple[bool, ...]:
        """Which of the 128 elements are active (``ARCH-stage-aaq-str.md`` §3.2)."""
        g = self.group_size
        return tuple(
            i // g < self.valid_partitions and i % g < self.valid_elements for i in range(LANES)
        )


def ds(
    valid_elements: int = LANES,
    num_partitions: int = 1,
    valid_partitions: int | None = None,
    mask_type: int = MaskType.ZERO,
) -> int:
    """The DS half for a vector shape; ``valid_partitions`` defaults to all groups."""
    shape = DataStructure(
        valid_elements=valid_elements,
        num_partitions=num_partitions,
        mask_type=MaskType(mask_type),
        valid_partitions=num_partitions if valid_partitions is None else valid_partitions,
    )
    f = DS_LAYOUT
    return (
        f["valid_elements"].put(shape.valid_elements)
        | f["num_partitions"].put(PARTITIONS.index(shape.num_partitions))
        | f["mask_type"].put(int(shape.mask_type))
        | f["valid_partitions"].put(shape.valid_partitions)
    )


def decode_ds(value: int) -> DataStructure:
    """The shape a DS half names; raises ``IsaValueError`` if it is invalid."""
    f = DS_LAYOUT
    return DataStructure(
        valid_elements=f["valid_elements"].get(value),
        num_partitions=PARTITIONS[f["num_partitions"].get(value)],
        mask_type=MaskType(f["mask_type"].get(value)),
        valid_partitions=f["valid_partitions"].get(value),
    )


#: The DS a state starts with in ``CR15.L``: all 128 elements, one group.
DEFAULT_DS: Final = ds()


# --- MULT constant -----------------------------------------------------------


@dataclass(frozen=True)
class Constant:
    """A decoded MULT constant: one signed or unsigned 8-bit NX8 element."""

    value: int
    scale: int
    fmt: Nx8Format


def constant(value: int, scale: int, exp_format: int, sign: int = 1) -> int:
    """The 32-bit constant CR for one NX8 element (always 8 bits, never dynamic)."""
    Nx8Format(sign=sign, exp_format=exp_format, num_bits=8)  # validates
    f = CONSTANT_LAYOUT
    return (
        f["value"].put(value)
        | f["scale"].put(scale)
        | f["exp_format"].put(exp_format)
        | f["sign"].put(sign)
    )


def decode_constant(word: int) -> Constant:
    """The element a constant CR holds; raises ``IsaValueError`` if its type is invalid."""
    f = CONSTANT_LAYOUT
    return Constant(
        value=f["value"].get(word),
        scale=f["scale"].get(word),
        fmt=Nx8Format(sign=f["sign"].get(word), exp_format=f["exp_format"].get(word), num_bits=8),
    )
