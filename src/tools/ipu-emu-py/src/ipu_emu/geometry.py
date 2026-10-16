"""Lane and row geometry the engine and its observers share.

Pure functions of the element format and of register contents. The engine's
handlers call them, and so do the observers (the alias profiler), so an
observer never reaches into ``Ipu`` for a helper (E04-11):

- :func:`element_format` names the active :class:`~ipu_isa.formats.ElementFormat`
  of a state (until E04-17 makes the format the state's own field);
- :func:`element_width_bytes` and :func:`row_size_bytes` give the bytes of one
  element and of one XMEM row in a format; :func:`xmem_element_width_bytes`
  and :func:`xmem_row_size_bytes` are the same for a state, on the hot path;
- :func:`lrd_lr_indices` names the two LRs behind an ``LrdIdx`` operand;
- :func:`effective_mult_mask` is the 128-bit lane mask a MULT applies;
- :func:`cr_scalar` is a CR's low byte as a signed element (the wide-vector
  scalar multiplicand).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

from ipu_emu.errors import EmulatorError
from ipu_isa.constants import LR_CR_SCALAR_BITS, Partition
from ipu_isa.formats import FORMATS, ElementFormat
from ipu_isa.geometry import LANES
from ipu_isa.registers import lrd_pair

if TYPE_CHECKING:
    from ipu_emu.ipu_state import IpuState
    from ipu_isa.constants import DStructureConfig

__all__ = [
    "cr_scalar",
    "effective_mult_mask",
    "element_format",
    "element_width_bytes",
    "inverse_partition_vector",
    "lrd_lr_indices",
    "partition_vector",
    "row_size_bytes",
    "xmem_element_width_bytes",
    "xmem_row_size_bytes",
]

#: All 128 lanes active.
_ALL_LANES = (1 << LANES) - 1

#: R_MASK holds eight 128-bit mask slots of 16 bytes each.
_MASK_SLOT_BYTES = LANES // 8
_MASK_SLOTS = LANES // _MASK_SLOT_BYTES

_NARROW_ELEMENT_BYTES = FORMATS[ElementFormat.INT8].element_bytes
_WIDE_ELEMENT_BYTES = FORMATS[ElementFormat.FP32].element_bytes


class _RegisterReader(Protocol):
    """The register reads these helpers need (a ``RegFile`` or its snapshot)."""

    def get_cr(self, index: int) -> int: ...

    def get_r_mask(self) -> bytes: ...


def element_format(state: IpuState) -> ElementFormat:
    """The element format *state* runs in.

    Wide-vector debug mode is FP32 or INT32 (its arithmetic); otherwise the
    format is the 8-bit one ``state.dtype`` names (``DType(x)`` is ``Ex``).

    Args:
        state: The machine state.

    Returns:
        Its element format.
    """
    if state.wide_vector_debug:
        return ElementFormat(state.wide_vector_arithmetic.value)
    return ElementFormat[state.dtype.name]


def element_width_bytes(fmt: ElementFormat) -> int:
    """Bytes of one element in *fmt*: 1 for an 8-bit format, 4 for FP32 and INT32."""
    return FORMATS[fmt].element_bytes


def row_size_bytes(fmt: ElementFormat) -> int:
    """Bytes of one XMEM row (LANES elements) in *fmt*: 128 or 512."""
    return FORMATS[fmt].row_bytes


def xmem_element_width_bytes(state: IpuState) -> int:
    """:func:`element_width_bytes` of *state*'s format, without building the enum."""
    return _WIDE_ELEMENT_BYTES if state.wide_vector_debug else _NARROW_ELEMENT_BYTES


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

    The wide-vector scalar multiplicand until E04-28 decodes CR scalars in the
    active format.
    """
    b = regfile.get_cr(cr_idx) & 0xFF
    return b if b < 128 else b - 256
