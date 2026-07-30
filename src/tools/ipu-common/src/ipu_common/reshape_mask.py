"""Lane-count mask for RESHAPE (issue #192).

RESHAPE permutes up to 8 R_ACC word lanes per instruction, addressed via the
8 byte lanes of a source/dest ``LrdIdx`` pair. ``reshape_mask`` limits how
many of those 8 lanes participate, encoded in ``RESHAPE_MASK_FIELD_BITS``
bits (no LR indirection) — same shape as ``mult_mask_offset``.
"""

RESHAPE_LANE_COUNT = 8
RESHAPE_MASK_FIELD_BITS = 3
