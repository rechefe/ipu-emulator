"""The two run modes' element formats and every size derived from them (ADR 0007).

A run is in one of two modes, named by its element format:

* ``NX8`` (HARDWARE): an XMEM row is 128 NX8 elements (128 bytes) plus one
  16-bit metadata word (``scale``, ``dynamic_exponent``). Which NX8 format a
  row holds (sign, ``fe``, width, dynamic) is not a run setting: each MULT and
  ``ACTIVATE.QUANTIZE`` names it with a QF half-CR (``ipu_isa.cr_halves``).
  The element codec is ``ipu_isa.nx8``.
* ``FP32`` (EMULATOR): a row is 128 IEEE binary32 elements (512 bytes), with
  no metadata, no scaling and no quantization. For debugging kernels that will
  run in NX8.

Each format is authored once as data in ``_FORMAT_SPEC`` and loaded into a
frozen ``FormatInfo``. Element bytes, row bytes, the metadata, the per-register
element layout and the addressable XMEM rows are derived from it. MULT_RES
and R_ACC hold FP32 in both modes (``AccumulatorType.FP32``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, TypedDict

from ipu_isa.errors import IsaLookupError
from ipu_isa.geometry import LANES, XMEM_BYTES
from ipu_isa.registers import REGISTER_DEFINITIONS
from ipu_isa.schema import SpecError
from ipu_isa.tags import Tag


class ElementFormat(str, Enum):
    """A run mode's element format. ``ElementFormat.NX8 == "nx8"``.

    A ``str`` mixin rather than ``StrEnum`` (Python 3.11), as for ``Tag``.
    """

    FP32 = "fp32"
    NX8 = "nx8"

    def __str__(self) -> str:
        return self.value


class AccumulatorType(str, Enum):
    """The type MULT_RES and R_ACC hold (see GLOSSARY.md): FP32 in both modes."""

    FP32 = "fp32"

    def __str__(self) -> str:
        return self.value


#: Bits of one accumulator element, per accumulator type.
ACCUMULATOR_BITS: Mapping[AccumulatorType, int] = MappingProxyType({AccumulatorType.FP32: 32})

#: Bits of one row's metadata word in NX8 mode: ``scale[7:0]``,
#: ``dynamic_exponent[8]`` (ADR 0007; the layout is in the emulator's XMEM).
NX8_METADATA_BITS: Final = 16


@dataclass(frozen=True, slots=True)
class FormatInfo:
    """Everything the ISA states about one run mode's element format."""

    format: ElementFormat
    tag: Tag  # HARDWARE: NX8; EMULATOR: FP32
    element_bits: int  # 8 or 32
    metadata_bits: int  # bits of row metadata: 16 for NX8, 0 for FP32
    accumulator: AccumulatorType  # FP32 in both modes

    @property
    def element_bytes(self) -> int:
        """Bytes of one element."""
        return self.element_bits // 8

    @property
    def row_bytes(self) -> int:
        """Bytes of one XMEM row: one element per lane (metadata not included)."""
        return LANES * self.element_bytes

    @property
    def has_metadata(self) -> bool:
        """True when a row carries a metadata word (NX8)."""
        return self.metadata_bits > 0

    @property
    def accumulator_bytes(self) -> int:
        """Bytes of one MULT_RES / R_ACC element."""
        return ACCUMULATOR_BITS[self.accumulator] // 8


class _FormatEntry(TypedDict):
    tag: Tag
    element_bits: int
    metadata_bits: int
    accumulator: AccumulatorType


# The authored data.
_FORMAT_SPEC: Final[Mapping[ElementFormat, _FormatEntry]] = {
    ElementFormat.FP32: {
        "tag": Tag.EMULATOR,
        "element_bits": 32,
        "metadata_bits": 0,
        "accumulator": AccumulatorType.FP32,
    },
    ElementFormat.NX8: {
        "tag": Tag.HARDWARE,
        "element_bits": 8,
        "metadata_bits": NX8_METADATA_BITS,
        "accumulator": AccumulatorType.FP32,
    },
}


def validate_formats(formats: Mapping[ElementFormat, FormatInfo]) -> None:
    """Raise ``SpecError`` if a format breaks a schema rule.

    Every format is listed once with exactly one tag (HARDWARE or EMULATOR);
    an element is a whole number of bytes; the accumulator is FP32.
    """
    missing = set(ElementFormat) - set(formats)
    if missing:
        raise SpecError(f"formats: missing {sorted(str(f) for f in missing)}")
    for fmt, info in formats.items():
        path = f"formats.{fmt}"
        if info.format is not fmt:
            raise SpecError(f"{path}: keyed under {fmt} but describes {info.format}")
        if info.tag not in (Tag.HARDWARE, Tag.EMULATOR):
            raise SpecError(f"{path}.tag: {info.tag} is not HARDWARE or EMULATOR")
        if info.element_bits <= 0 or info.element_bits % 8:
            raise SpecError(f"{path}.element_bits: {info.element_bits} is not whole bytes")
        if info.accumulator is not AccumulatorType.FP32:
            raise SpecError(f"{path}.accumulator: {info.accumulator}, expected fp32")


#: Every element format, in enum order.
FORMATS: Mapping[ElementFormat, FormatInfo] = MappingProxyType(
    {fmt: FormatInfo(format=fmt, **_FORMAT_SPEC[fmt]) for fmt in ElementFormat}
)
validate_formats(FORMATS)


# How each register's elements depend on the format. Element registers hold
# one element per byte of their NX8 allocation, at the format's width;
# accumulator registers hold one accumulator element per lane; R_MASK holds
# lane-mask bits and is the same in every format.
_ELEMENT_REGISTERS: Final = frozenset({"r", "r_cyclic", "mem_bypass"})
_ACCUMULATOR_REGISTERS: Final = frozenset({"mult_res", "r_acc"})
_MASK_REGISTERS: Final = frozenset({"r_mask"})


def register_layout(register: str, fmt: ElementFormat) -> tuple[int, int]:
    """Return ``(elements, element_bytes)`` of one ``register`` in format ``fmt``.

    ``register`` is a ``REGISTER_DEFINITIONS`` key, which holds the NX8 sizes.
    """
    info = FORMATS[ElementFormat(fmt)]
    if register not in REGISTER_DEFINITIONS:
        raise IsaLookupError(f"unknown register {register!r}")
    size_bytes = REGISTER_DEFINITIONS[register]["size_bytes"]
    if register in _ELEMENT_REGISTERS:
        return size_bytes, info.element_bytes
    if register in _ACCUMULATOR_REGISTERS:
        return LANES, info.accumulator_bytes
    if register in _MASK_REGISTERS:
        return size_bytes, 1
    raise IsaLookupError(f"register {register!r} has no format-dependent element layout")


def addressable_rows(fmt: ElementFormat) -> int:
    """XMEM rows a program may address in ``fmt``.

    The byte address space (``XMEM_BYTES``) is the same in both modes; the row
    count is that space divided by the row width (metadata is a side store and
    takes no data bytes).
    """
    return XMEM_BYTES // FORMATS[ElementFormat(fmt)].row_bytes
