"""Mask slot selector for multiply masking (issue #46).

``r_mask`` holds eight 128-bit slots (128 bytes / 16 bytes each). The mult-stage
``mask_offset`` operand selects the slot using an immediate in ``[0, 7]``
encoded in ``MULT_MASK_OFFSET_FIELD_BITS`` bits (no LR indirection).
"""

MULT_MASK_SLOT_COUNT = 8
MULT_MASK_OFFSET_FIELD_BITS = 3
