"""Lane and row geometry the engine and its observers share.

Pure functions of the element format and of register contents. The engine's
handlers call them, and so do the observers (the alias profiler), so an
observer never reaches into ``Ipu`` for a helper (E04-11):

- a state's element format is its own field (``IpuState.fmt``, E04-17), and
  ``IpuState.codec`` is that format's lane codec;
- :func:`vector_registers` names the registers that hold Ra (R0 ++ R1) and
  R_CYCLIC in a state's element width; :data:`WIDE_REGISTERS` maps each
  declared register to its emulator-only 4-byte storage and
  :data:`EMULATOR_REGISTERS` is the set of those (the one list of them);
- :func:`is_wide_format` says whether a format is one of the 4-byte ones;
- :func:`element_width_bytes` and :func:`row_size_bytes` give the bytes of one
  element and of one XMEM row in a format; :func:`xmem_element_width_bytes`
  and :func:`xmem_row_size_bytes` are the same for a state, on the hot path;
- :func:`lrd_lr_indices` names the two LRs behind an ``LrdIdx`` operand;
- :func:`effective_mult_mask` is the 128-bit lane mask a MULT applies;
- :func:`cr_scalar` is a CR as a MULT constant in the state's mode (the
  MULT handlers' scalar and the alias observer's identity check);
- :func:`xmem_row_addr` turns an assembly row number into a byte address;
- :func:`lrd_bytes` reads the eight byte elements of an LRD pair;
- :func:`lane_format` is the struct format of one 4-byte accumulator lane
  (the AAQ activation path's; every other handler goes through the codec);
- :func:`agg_lane_count` is how many R_ACC lanes a reduction includes;
- ``R_REG_SIZE``, ``R_CYCLIC_SIZE`` and ``R_CYCLIC_VALID_INDICES`` are the R
  register and R_CYCLIC sizes in bytes and R_CYCLIC's slot boundaries.

The slot modules (``ipu_emu.slots``) share these instead of importing each
other (E04-14).
"""

from __future__ import annotations

from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol

from ipu_emu.cr_contents import constant_value, fp32_constant_value
from ipu_emu.errors import EmulatorError
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.constants import LR_CR_SCALAR_BITS
from ipu_isa.cr_halves import PARTITIONS
from ipu_isa.faults import FaultId
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.registers import REGISTER_DEFINITIONS, get_register_sizes, lrd_pair

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ipu_emu.ipu_state import IpuState
    from ipu_isa.cr_halves import DataStructure

__all__ = [
    "EIGHT_BIT_ROWS",
    "EMULATOR_REGISTERS",
    "R_CYCLIC_SIZE",
    "R_CYCLIC_VALID_INDICES",
    "R_REG_SIZE",
    "WIDE_REGISTERS",
    "VectorRegisters",
    "agg_lane_count",
    "cr_scalar",
    "effective_mult_mask",
    "element_width_bytes",
    "fp32_mode",
    "inverse_partition_vector",
    "is_wide_format",
    "lane_format",
    "lrd_bytes",
    "lrd_lr_indices",
    "partition_vector",
    "row_size_bytes",
    "vector_registers",
    "xmem_element_width_bytes",
    "xmem_row_addr",
    "xmem_row_check",
    "xmem_row_fault",
    "xmem_row_size_bytes",
]

_REG_SIZES = get_register_sizes()

#: Bytes of one R register (one row in an 8-bit format) and of the R_CYCLIC ring, from
#: ``REGISTER_DEFINITIONS`` via ``get_register_sizes()``.
R_REG_SIZE = _REG_SIZES["r"]["size_bytes"]
R_CYCLIC_SIZE = _REG_SIZES["r_cyclic"]["size_bytes"]

#: R_CYCLIC is divided into four 128-byte slots; LD.MULT.CYCLIC's index must
#: land exactly on a slot boundary (no implicit wraparound).
R_CYCLIC_VALID_INDICES = tuple(range(0, R_CYCLIC_SIZE, R_REG_SIZE))

# XMEM is allocated at XMEM_SIZE_BYTES in both modes. NX8 may address only the
# first 16384 rows of 128 B (2 MB) -- a fixed limit, not derived from the
# allocation, so enlarging XMEM for the FP32 kernels does not silently widen
# what NX8 programs can reach. FP32 reaches the whole allocation
# (XMEM_ADDRESSABLE_ROWS rows of 512 B).
EIGHT_BIT_ROWS = 16384

#: All 128 lanes active.
_ALL_LANES = (1 << LANES) - 1

#: R_MASK holds eight 128-bit mask slots of 16 bytes each.
_MASK_SLOT_BYTES = LANES // 8
_MASK_SLOTS = LANES // _MASK_SLOT_BYTES

_EIGHT_BIT_ELEMENT_BYTES = FORMATS[ElementFormat.NX8].element_bytes

#: The name suffix of a register's emulator-only 4-byte storage (``ipu_isa.registers``).
_WIDE_SUFFIX = "_wide_debug"

#: Declared register -> the emulator-only register that holds it in the FP32
#: mode: ``r`` -> ``r_wide_debug``, ``r_cyclic`` ->
#: ``r_cyclic_wide_debug``. Derived from ``REGISTER_DEFINITIONS``; the one
#: place in ``ipu_emu`` that names these registers.
WIDE_REGISTERS: Mapping[str, str] = MappingProxyType(
    {
        name.removesuffix(_WIDE_SUFFIX): name
        for name in REGISTER_DEFINITIONS
        if name.endswith(_WIDE_SUFFIX)
    }
)

#: Registers with no hardware equivalent: the FP32 mode's storage.
EMULATOR_REGISTERS: frozenset[str] = frozenset(WIDE_REGISTERS.values())


class _RegisterReader(Protocol):
    """The register reads these helpers need (a ``RegFile`` or its snapshot)."""

    def get_cr(self, index: int) -> int: ...

    def get_lr(self, index: int) -> int: ...

    def get_r_mask(self) -> bytes: ...


def is_wide_format(fmt: ElementFormat) -> bool:
    """*fmt* is the 4-byte element format (FP32, emulator-only)."""
    return FORMATS[fmt].element_bytes != _EIGHT_BIT_ELEMENT_BYTES


def _is_wide(state: IpuState) -> bool:
    """*state* runs in the 4-byte element format (FP32, emulator-only)."""
    return state.codec.element_bytes != _EIGHT_BIT_ELEMENT_BYTES


def fp32_mode(state: IpuState) -> bool:
    """*state* runs in FP32 mode (emulator-only): rows of 512 B, no metadata,
    no quantization, QF ignored (ADR 0007). The one mode test the slot
    handlers make."""
    return _is_wide(state)


class VectorRegisters(NamedTuple):
    """The register names behind the MULT stage's vector operands in one element width."""

    #: R0 ++ R1 (Ra): two rows of LANES elements, R0 first.
    ra: str
    #: The R_CYCLIC ring: 512 elements.
    r_cyclic: str


_EIGHT_BIT_REGISTERS = VectorRegisters(ra="r", r_cyclic="r_cyclic")
_WIDE_REGISTERS = VectorRegisters(
    ra=WIDE_REGISTERS[_EIGHT_BIT_REGISTERS.ra],
    r_cyclic=WIDE_REGISTERS[_EIGHT_BIT_REGISTERS.r_cyclic],
)


def vector_registers(state: IpuState) -> VectorRegisters:
    """Where Ra and R_CYCLIC live in *state*'s element width.

    The FP32 mode keeps its elements in their own registers
    (``r_wide_debug``, ``r_cyclic_wide_debug``); both layouts hold the same
    number of elements, so a handler indexes either in elements.
    """
    return _WIDE_REGISTERS if _is_wide(state) else _EIGHT_BIT_REGISTERS


def element_width_bytes(fmt: ElementFormat) -> int:
    """Bytes of one element in *fmt*: 1 for NX8, 4 for FP32."""
    return FORMATS[fmt].element_bytes


def row_size_bytes(fmt: ElementFormat) -> int:
    """Bytes of one XMEM row (LANES elements) in *fmt*: 128 or 512."""
    return FORMATS[fmt].row_bytes


def xmem_element_width_bytes(state: IpuState) -> int:
    """:func:`element_width_bytes` of *state*'s format, from its codec (the hot path)."""
    return state.codec.element_bytes


def xmem_row_size_bytes(state: IpuState) -> int:
    """:func:`row_size_bytes` of *state*'s format: one assembly-addressable XMEM row."""
    return LANES * xmem_element_width_bytes(state)


def lrd_lr_indices(n: int) -> tuple[int, int]:
    """The (low, high) LR indices of ``LrdIdx`` operand value *n* (LRD{2n})."""
    return lrd_pair(n)


def _check_partition(num_partitions: object) -> None:
    # Reached mid-run from the MULT mask path: an internal invariant, not a setup error.
    if isinstance(num_partitions, bool) or num_partitions not in PARTITIONS:
        raise EmulatorError(f"num_partitions must be one of {PARTITIONS}, got {num_partitions!r}")


@cache
def partition_vector(num_partitions: int) -> int:
    """The left-shift partition vector: 0 at the START of each group.

    Used for positive mask shifts (+1, +2, +3). One group is all ones (no
    boundaries); P groups of LANES/P elements have bit 0 of each group clear.

    Raises:
        EmulatorError: *num_partitions* is not 1, 2, 4 or 8.
    """
    _check_partition(num_partitions)
    if num_partitions == 1:
        return _ALL_LANES
    step = LANES // num_partitions
    return sum(1 << i for i in range(LANES) if i % step != 0)


@cache
def inverse_partition_vector(num_partitions: int) -> int:
    """The right-shift partition vector: 0 at the END of each group.

    Used for negative mask shifts (-1, -2, -3). One group is all ones.

    Raises:
        EmulatorError: *num_partitions* is not 1, 2, 4 or 8.
    """
    _check_partition(num_partitions)
    if num_partitions == 1:
        return _ALL_LANES
    step = LANES // num_partitions
    return sum(1 << i for i in range(LANES) if i % step != step - 1)


def effective_mult_mask(
    regfile: _RegisterReader, shape: DataStructure, mask_idx: int, shift: int
) -> int:
    """The lane mask a MULT applies: an R_MASK slot, shifted within partitions.

    The slot is ``mask_idx`` modulo the eight slots. ``shift`` is an LR value,
    sign-extended from ``LR_CR_SCALAR_BITS`` and clamped to [-3, +3]; each step
    shifts by one lane and ANDs with the partition vector of
    ``shape.num_partitions`` (the inverse vector for right shifts).

    Args:
        regfile: The register file whose R_MASK is read.
        shape: The MULT's DS.
        mask_idx: The ``mask_offset`` operand.
        shift: The ``mask_shift`` operand's LR value.

    Returns:
        The 128-bit mask, bit i set when lane i is active.
    """
    if shift >= (1 << (LR_CR_SCALAR_BITS - 1)):
        shift -= 1 << LR_CR_SCALAR_BITS
    shift = max(-3, min(3, shift))
    offset = (mask_idx % _MASK_SLOTS) * _MASK_SLOT_BYTES
    mask_bytes = regfile.get_r_mask()
    mask = int.from_bytes(mask_bytes[offset : offset + _MASK_SLOT_BYTES], "little") & _ALL_LANES
    if shift < 0:
        pv = inverse_partition_vector(shape.num_partitions)
        for _ in range(-shift):
            mask = (mask >> 1) & pv
    elif shift > 0:
        pv = partition_vector(shape.num_partitions)
        for _ in range(shift):
            mask = (mask << 1) & pv & _ALL_LANES
    return mask


def cr_scalar(state: IpuState, cr_idx: int, word: int | None = None) -> float:
    """CR *cr_idx* as a MULT constant in *state*'s mode (ADR 0007).

    ``CR0`` is 0.0 and ``CR1`` 1.0 in both modes. Any other CR is, in NX8,
    one NX8 element in the constant layout (``ipu_isa.cr_halves``) and, in
    FP32, its IEEE float32 bits: what a MULT with this CR as its scalar
    operand multiplies by.

    Args:
        state: The state, for its mode.
        cr_idx: The CR.
        word: The CR's value as the reading instruction sees it; the live
            register when omitted.

    Raises:
        CrContentError: NX8 mode and the CR holds no valid constant.
    """
    value = state.regfile.get_cr(cr_idx) if word is None else word
    if _is_wide(state):
        return fp32_constant_value(cr_idx, value)
    return constant_value(cr_idx, value)


def xmem_row_fault(state: IpuState, row: int) -> str | None:
    """Why *row* is no XMEM row of the active mode (fault F-6), or ``None``.

    A pure check of the row itself; :func:`xmem_row_addr` raises on it, and
    the precise-fault check phase (E04-20) evaluates it before the word runs.
    """
    if row < 0:
        return f"XMEM row must be non-negative; got {row}"
    if not _is_wide(state) and row >= EIGHT_BIT_ROWS:
        return (
            f"XMEM row {row} is out of range in NX8 mode "
            f"(rows 0..{EIGHT_BIT_ROWS - 1}, the first "
            f"{EIGHT_BIT_ROWS * xmem_row_size_bytes(state) >> 20} MB of XMEM)"
        )
    addr = row * xmem_row_size_bytes(state)
    if addr >= XMEM_SIZE_BYTES:
        return (
            f"XMEM row {row} is out of range: byte address {addr} exceeds "
            f"the {XMEM_SIZE_BYTES}-byte allocation"
        )
    return None


def xmem_row_check(state: IpuState, row: int) -> tuple[FaultId, str] | None:
    """:func:`xmem_row_fault` as a fault predicate's result: F-6 and why."""
    detail = xmem_row_fault(state, row)
    return None if detail is None else (FaultId.F6, detail)


def xmem_row_addr(state: IpuState, row: int) -> int:
    """Translate an XMEM row number to a byte address in the active element format.

    ``.asm`` XMEM operands (``offset + base``) are row numbers, not byte
    addresses — one row is LANES elements, so the same row number reaches
    the same logical row in every format at different byte offsets. XMEM is
    allocated at XMEM_SIZE_BYTES unconditionally; an 8-bit format may only
    *address* the first EIGHT_BIT_ROWS rows (2 MB) of that allocation.

    This only translates and range-checks the row itself; the resulting
    address's actual payload is bounds-checked by
    ``XMem.read_address``/``write_address`` against the allocation.

    The row is legal: the word's check phase faulted F-6 on any other
    (:func:`xmem_row_fault`, E04-20).

    Raises:
        EmulatorError: The row is not legal: a handler ran on a word no
            check phase saw (a direct call).
    """
    detail = xmem_row_fault(state, row)
    if detail is not None:
        raise EmulatorError(f"{detail} (an unchecked word)")
    return row * xmem_row_size_bytes(state)


def lrd_bytes(regfile: _RegisterReader, n: int) -> bytes:
    """Read LRDn's 8 byte elements from a register file (snapshot or live).

    Elements 0-3 are LR(2n)'s bytes (little-endian), elements 4-7 are LR(2n+1)'s.
    """
    lo_idx, hi_idx = lrd_pair(n)
    return regfile.get_lr(lo_idx).to_bytes(4, "little") + regfile.get_lr(hi_idx).to_bytes(
        4, "little"
    )


def lane_format(state: IpuState) -> str:
    """Struct format of one 4-byte R_ACC / MULT_RES / AGG lane: FP32 in both modes."""
    return "<f"


def agg_lane_count(valid_elements: int) -> int:
    """Number of R_ACC words a reduction includes (clamped to LANES)."""
    v = int(valid_elements) & 0xFFFFFFFF
    return min(v, LANES)
