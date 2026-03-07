"""Single source of truth for acc.stride operand enums.

Used by:
- ipu-as: enum names for assembly syntax (ElementsInRowField, HorizontalStrideField, VerticalStrideField)
- ipu-emu: interpretation of encoded values in execute_acc_stride

Encoding convention:
- elements_in_row: index 0..3 → 8, 16, 32, 64 elements per row.
- horizontal_stride: 3 bits → (enabled, inverted, expand) = (bit0, bit1, bit2).
- vertical_stride: 2 bits → (enabled, inverted) = (bit0, bit1).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Elements per row (acc.stride first operand)
# ---------------------------------------------------------------------------

ELEMENTS_IN_ROW_NAMES: tuple[str, ...] = ("8", "16", "32", "64")
ELEMENTS_IN_ROW_VALUES: tuple[int, ...] = (8, 16, 32, 64)


def get_elements_per_row(encoded: int) -> int:
    """Return elements per row for encoded value 0..3."""
    return ELEMENTS_IN_ROW_VALUES[encoded & 3]


# ---------------------------------------------------------------------------
# Horizontal stride (enabled=bit0, inverted=bit1, expand=bit2)
# ---------------------------------------------------------------------------

HORIZONTAL_STRIDE_NAMES: tuple[str, ...] = (
    "off",           # 0: disabled
    "on",            # 1: enabled, not inverted, no expand
    "on_inv",        # 2: enabled, inverted, no expand
    "on_expand",     # 3: enabled, not inverted, expand
    "on_inv_expand", # 4: enabled, inverted, expand
    "reserved5",
    "reserved6",
    "reserved7",
)


def get_horizontal_stride_bits(encoded: int) -> tuple[bool, bool, bool]:
    """Return (enabled, inverted, expand) for encoded horizontal stride 0..7."""
    return (
        bool(encoded & 1),
        bool(encoded & 2),
        bool(encoded & 4),
    )


# ---------------------------------------------------------------------------
# Vertical stride (enabled=bit0, inverted=bit1)
# ---------------------------------------------------------------------------

VERTICAL_STRIDE_NAMES: tuple[str, ...] = (
    "off",        # 0: disabled
    "on",         # 1: enabled, not inverted
    "on_inv",     # 2: enabled, inverted
    "reserved3",
)


def get_vertical_stride_bits(encoded: int) -> tuple[bool, bool]:
    """Return (enabled, inverted) for encoded vertical stride 0..3."""
    return (
        bool(encoded & 1),
        bool(encoded & 2),
    )
