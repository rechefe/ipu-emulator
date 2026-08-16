"""RESHAPE ``reshape_mask`` operand — bit width derived from union layout (issue #192, #210).

RESHAPE copies elements from MULT_RES into R_ACC via two LrdIdx byte-index
arrays. ``reshape_mask`` selects the TRAILING (RESHAPE_ELEMENT_COUNT - mask)
elements of each LRD pair to participate (elements mask..RESHAPE_ELEMENT_COUNT-1).

The field supports both an immediate (0..RESHAPE_ELEMENT_COUNT-1, encoded as
itself) and an LR register (encoded as LR_index + RESHAPE_MASK_LR_OFFSET). The
field width is not hardcoded: after ``compute_slot_layouts`` runs, the union
solver writes ``RESHAPE_MASK_FIELD_BITS`` from the shared union field that
carries the ``LrOrReshapeMaskImmediate`` operand (same bin as
``DstructureCrIdx``/``ElementsInRow`` in the ``acc`` slot). The number of LR
registers encodable is therefore whatever room remains above
RESHAPE_MASK_LR_OFFSET.
"""

RESHAPE_ELEMENT_COUNT = 8
RESHAPE_MASK_LR_OFFSET = 8

RESHAPE_MASK_FIELD_BITS: int = 0


def reshape_mask_field_max() -> int:
    """Return the maximum encodable value (inclusive) of the reshape_mask field."""
    if RESHAPE_MASK_FIELD_BITS <= 0:
        raise RuntimeError(
            "RESHAPE_MASK_FIELD_BITS is unset; union layout must be computed first"
        )
    return (1 << RESHAPE_MASK_FIELD_BITS) - 1
