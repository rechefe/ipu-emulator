"""RESHAPE ``reshape_mask`` operand — bit width derived from union layout (issue #192, #210).

RESHAPE copies elements from MULT_RES into R_ACC via two LrdIdx byte-index
arrays. ``reshape_mask`` selects the TRAILING (RESHAPE_ELEMENT_COUNT - mask)
elements of each LRD pair to participate (elements mask..RESHAPE_ELEMENT_COUNT-1).

The field supports both an immediate (0..RESHAPE_ELEMENT_COUNT-1, encoded as
itself) and an LR register (encoded as LR_index + RESHAPE_MASK_LR_OFFSET). The
field width is not hardcoded: the union solver packs the
``LrOrReshapeMaskImmediate`` operand into a shared ``acc`` field (same bin as
``HalfCrIdx``/``ElementsInRow``), and :func:`reshape_mask_field_bits`
reads that field's width from :func:`ipu_isa.layout.isa_layout`. The number of
LR registers encodable is therefore whatever room remains above
RESHAPE_MASK_LR_OFFSET.
"""

from __future__ import annotations

from ipu_isa.layout import isa_layout

RESHAPE_ELEMENT_COUNT = 8
RESHAPE_MASK_LR_OFFSET = 8


def reshape_mask_field_bits() -> int:
    """Width of the reshape_mask field."""
    return isa_layout().operand_bits["LrOrReshapeMaskImmediate"]


def reshape_mask_field_max() -> int:
    """Return the maximum encodable value (inclusive) of the reshape_mask field."""
    return (1 << reshape_mask_field_bits()) - 1
