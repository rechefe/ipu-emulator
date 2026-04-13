"""Single source of truth for acc.stride operand enums.

Used by:
- ipu-as: enum names for assembly syntax (ElementsInRowField, HorizontalStrideField, VerticalStrideField)
- ipu-emu: interpretation of encoded values in execute_acc_stride

Encoding convention:
- elements_in_row: index 0..3 → 8, 16, 32, 64 elements per row.
- horizontal_stride: semantic enum 0..4 → (enabled, inverted, expand) via lookup table.
- vertical_stride: semantic enum 0..2 → (enabled, inverted) via lookup table.

Note: horizontal/vertical stride values are NOT a packed bit-field; they are a
sequential enum. Decoding is done via explicit lookup tables to avoid silent
mis-decoding (e.g. treating encoded=2 as enabled=False because bit0 is clear).
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
# Horizontal stride (semantic enum: 0=off, 1=on, 2=on_inv, 3=on_expand, 4=on_inv_expand)
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


_HORIZONTAL_STRIDE_DECODE: dict[int, tuple[bool, bool, bool]] = {
    0: (False, False, False),  # off
    1: (True,  False, False),  # on
    2: (True,  True,  False),  # on_inv
    3: (True,  False, True),   # on_expand
    4: (True,  True,  True),   # on_inv_expand
}


def get_horizontal_stride_bits(encoded: int) -> tuple[bool, bool, bool]:
    """Return (enabled, inverted, expand) for encoded horizontal stride 0..4."""
    return _HORIZONTAL_STRIDE_DECODE[encoded]


# ---------------------------------------------------------------------------
# Vertical stride (semantic enum: 0=off, 1=on, 2=on_inv)
# ---------------------------------------------------------------------------

VERTICAL_STRIDE_NAMES: tuple[str, ...] = (
    "off",        # 0: disabled
    "on",         # 1: enabled, not inverted
    "on_inv",     # 2: enabled, inverted
    "reserved3",
)


_VERTICAL_STRIDE_DECODE: dict[int, tuple[bool, bool]] = {
    0: (False, False),  # off
    1: (True,  False),  # on
    2: (True,  True),   # on_inv
}


def get_vertical_stride_bits(encoded: int) -> tuple[bool, bool]:
    """Return (enabled, inverted) for encoded vertical stride 0..2."""
    return _VERTICAL_STRIDE_DECODE[encoded]
