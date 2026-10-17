"""Lane and row geometry the engine and its observers share.

Pure functions of the element format and of register contents. The engine's
handlers call them, and so do the observers (the alias profiler), so an
observer never reaches into ``Ipu`` for a helper (E04-11):

- a state's element format is its own field (``IpuState.fmt``, E04-17), and
  ``IpuState.codec`` is that format's lane codec;
- :func:`vector_registers` names the registers that hold Ra (R0 ++ R1) and
  R_CYCLIC in a state's element width;
- :func:`element_width_bytes` and :func:`row_size_bytes` give the bytes of one
  element and of one XMEM row in a format; :func:`xmem_element_width_bytes`
  and :func:`xmem_row_size_bytes` are the same for a state, on the hot path;
- :func:`lrd_lr_indices` names the two LRs behind an ``LrdIdx`` operand;
- :func:`effective_mult_mask` is the 128-bit lane mask a MULT applies;
- :func:`cr_scalar` is a CR's low byte as a signed element (the alias
  observer's identity check; the MULT handlers decode CR scalars through
  :meth:`~ipu_emu.numerics.LaneCodec.decode_scalar`);
- :func:`xmem_row_addr` turns an assembly row number into a byte address;
- :func:`lrd_bytes` reads the eight byte elements of an LRD pair;
- :func:`lane_format` is the struct format of one 4-byte accumulator lane
  (the AAQ activation path's; every other handler goes through the codec);
- :func:`agg_lane_count` is how many R_ACC lanes a reduction includes.

The slot modules (``ipu_emu.slots``) share these instead of importing each
other (E04-14).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple, Protocol

from ipu_emu.errors import EmulatorError
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_isa.constants import LR_CR_SCALAR_BITS, Partition
from ipu_isa.faults import FaultId
from ipu_isa.formats import FORMATS, AccumulatorType, ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.registers import lrd_pair

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState
    from ipu_isa.constants import DStructureConfig

__all__ = [
    "NARROW_MAX_ROW",
    "VectorRegisters",
    "agg_lane_count",
    "cr_scalar",
    "effective_mult_mask",
    "element_width_bytes",
    "inverse_partition_vector",
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

# XMEM is allocated at XMEM_SIZE_BYTES always (mode-independent). Narrow mode
# may address only its first 16384 rows of 128 B (2 MB) -- a fixed limit, not
# derived from the allocation, so enlarging XMEM for wide-mode kernels does not
# silently widen what narrow-mode programs can reach. Debug mode reaches the
# whole allocation (XMEM_ADDRESSABLE_ROWS rows of 512 B).
NARROW_MAX_ROW = 16384

#: All 128 lanes active.
_ALL_LANES = (1 << LANES) - 1

#: R_MASK holds eight 128-bit mask slots of 16 bytes each.
_MASK_SLOT_BYTES = LANES // 8
_MASK_SLOTS = LANES // _MASK_SLOT_BYTES

_NARROW_ELEMENT_BYTES = FORMATS[ElementFormat.INT8].element_bytes


class _RegisterReader(Protocol):
    """The register reads these helpers need (a ``RegFile`` or its snapshot)."""

    def get_cr(self, index: int) -> int: ...

    def get_lr(self, index: int) -> int: ...

    def get_r_mask(self) -> bytes: ...


def _is_wide(state: IpuState) -> bool:
    """*state* runs in a 4-byte element format (FP32 or INT32, emulator-only)."""
    return state.codec.element_bytes != _NARROW_ELEMENT_BYTES


class VectorRegisters(NamedTuple):
    """The register names behind the MULT stage's vector operands in one element width."""

    #: R0 ++ R1 (Ra): two rows of LANES elements, R0 first.
    ra: str
    #: The R_CYCLIC ring: 512 elements.
    r_cyclic: str


_NARROW_REGISTERS = VectorRegisters(ra="r", r_cyclic="r_cyclic")
_WIDE_REGISTERS = VectorRegisters(ra="r_wide_debug", r_cyclic="r_cyclic_wide_debug")


def vector_registers(state: IpuState) -> VectorRegisters:
    """Where Ra and R_CYCLIC live in *state*'s element width.

    A 4-byte format (FP32, INT32) keeps its elements in their own registers
    (``r_wide_debug``, ``r_cyclic_wide_debug``); both layouts hold the same
    number of elements, so a handler indexes either in elements.
    """
    return _WIDE_REGISTERS if _is_wide(state) else _NARROW_REGISTERS


def element_width_bytes(fmt: ElementFormat) -> int:
    """Bytes of one element in *fmt*: 1 for an 8-bit format, 4 for FP32 and INT32."""
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
    if not isinstance(num_partitions, Partition):
        raise EmulatorError(f"partition must be a Partition enum value, got {num_partitions!r}")


@lru_cache(maxsize=None, typed=True)  # typed: a plain int must still raise
def partition_vector(num_partitions: Partition) -> int:
    """The left-shift partition vector: 0 at the START of each group.

    Used for positive mask shifts (+1, +2, +3). ``Partition`` 0 is all ones (no
    boundaries); P groups of LANES/P elements have bit 0 of each group clear.

    Raises:
        EmulatorError: *num_partitions* is not a ``Partition``.
    """
    _check_partition(num_partitions)
    if num_partitions == 0:
        return _ALL_LANES
    step = LANES // num_partitions
    return sum(1 << i for i in range(LANES) if i % step != 0)


@lru_cache(maxsize=None, typed=True)  # typed: a plain int must still raise
def inverse_partition_vector(num_partitions: Partition) -> int:
    """The right-shift partition vector: 0 at the END of each group.

    Used for negative mask shifts (-1, -2, -3). ``Partition`` 0 is all ones.

    Raises:
        EmulatorError: *num_partitions* is not a ``Partition``.
    """
    _check_partition(num_partitions)
    if num_partitions == 0:
        return _ALL_LANES
    step = LANES // num_partitions
    return sum(1 << i for i in range(LANES) if i % step != step - 1)


def effective_mult_mask(
    regfile: _RegisterReader, dstructure: DStructureConfig, mask_idx: int, shift: int
) -> int:
    """The lane mask a MULT applies: an R_MASK slot, shifted within partitions.

    The slot is ``mask_idx`` modulo the eight slots. ``shift`` is an LR value,
    sign-extended from ``LR_CR_SCALAR_BITS`` and clamped to [-3, +3]; each step
    shifts by one lane and ANDs with the partition vector of
    ``dstructure.partition`` (the inverse vector for right shifts).

    Args:
        regfile: The register file whose R_MASK is read.
        dstructure: The MULT's dstructure configuration.
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
        pv = inverse_partition_vector(dstructure.partition)
        for _ in range(-shift):
            mask = (mask >> 1) & pv
    elif shift > 0:
        pv = partition_vector(dstructure.partition)
        for _ in range(shift):
            mask = (mask << 1) & pv & _ALL_LANES
    return mask


def cr_scalar(regfile: _RegisterReader, cr_idx: int) -> int:
    """CR *cr_idx*'s low byte as a signed element (the CR itself is not widened).

    What ``Fp32Codec.decode_scalar`` and ``IntCodec.decode_scalar`` read as a
    number, until E04-28 decodes CR scalars in the active format.
    """
    b = regfile.get_cr(cr_idx) & 0xFF
    return b if b < 128 else b - 256


def xmem_row_fault(state: IpuState, row: int) -> str | None:
    """Why *row* is no XMEM row of the active mode (fault F-6), or ``None``.

    A pure check of the row itself; :func:`xmem_row_addr` raises on it, and
    the precise-fault check phase (E04-20) evaluates it before the word runs.
    """
    if row < 0:
        return f"XMEM row must be non-negative; got {row}"
    if not _is_wide(state) and row >= NARROW_MAX_ROW:
        return (
            f"XMEM row {row} is out of range for narrow mode "
            f"(rows 0..{NARROW_MAX_ROW - 1}, the first "
            f"{NARROW_MAX_ROW * xmem_row_size_bytes(state) >> 20} MB of XMEM)"
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
    """Translate an XMEM row number to a byte address in the active mode.

    ``.asm`` XMEM operands (``offset + base``) are row numbers, not byte
    addresses — one row is LANES elements, so the same row number reaches
    the same logical row in both modes at different byte offsets. XMEM is
    allocated at XMEM_SIZE_BYTES unconditionally; narrow mode may only
    *address* the first NARROW_MAX_ROW rows (2 MB) of that allocation.

    This only translates and range-checks the row itself; the resulting
    address's actual payload (which may span more than one row's worth of
    bytes, e.g. ST.ACC's fixed 512-byte R_ACC) is bounds-checked by
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
    """Struct format of one 4-byte R_ACC / MULT_RES / AGG lane: the format's accumulator."""
    return "<f" if state.format_info.accumulator is AccumulatorType.FP32 else "<i"


def agg_lane_count(valid_elements: int) -> int:
    """Number of R_ACC words a reduction includes (clamped to LANES)."""
    v = int(valid_elements) & 0xFFFFFFFF
    return min(v, LANES)
