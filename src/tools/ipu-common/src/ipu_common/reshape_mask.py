"""Element-count mask for RESHAPE (issue #192, issue #210).

RESHAPE copies elements from MULT_RES into R_ACC via two LrdIdx byte-index
arrays. ``reshape_mask`` selects the TRAILING (8 - reshape_mask) elements of
each LRD pair to participate (elements reshape_mask..7), encoded in
``RESHAPE_MASK_FIELD_BITS`` bits.

The field supports both an immediate (0-7, encoded as 0-7) and an LR register
(LR0-LR7, encoded as LR_index + RESHAPE_MASK_LR_OFFSET). At runtime the low 3
bits of the selected LR value are used as the mask.
"""

RESHAPE_ELEMENT_COUNT = 8
RESHAPE_MASK_FIELD_BITS = 4
RESHAPE_MASK_LR_OFFSET = 8
