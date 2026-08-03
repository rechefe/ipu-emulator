"""Single source of truth for ACC.DOWNSAMPLING operand enums.

Used by:
- ipu-as: enum names for assembly syntax (Stage1Field, Stage2Field, InvertField)
- ipu-emu: interpretation of encoded values in execute_downsampling

Pipeline (see instruction_spec.py for the full operand doc):
- stage1 selects alternating elements (even/odd) from the 128-element row, always
  producing 64 elements.
- stage2 keeps a subset of stage1's 64 elements, optionally inverted (stage2=64_128
  has no invert option — it passes all 64 elements through unchanged; 8_16 pads its
  32 taken elements back out to a full 64-wide span with interleaved zeros).
- The write footprint (32 or 64 elements) and the legal write-offset values both
  depend on stage2 — see get_write_width() / get_legal_offsets() / invert_allowed().
  The write offset itself comes from a live LR register (offset % 4) * 32, same
  convention as the pre-rename ACC.STRIDE instruction.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# stage1 — alternating-element selection from the 128-element row
# ---------------------------------------------------------------------------

STAGE1_NAMES: tuple[str, ...] = ("even", "odd")


def get_stage1_start(encoded: int) -> int:
    """Starting index (0 or 1) for encoded stage1 (0=even, 1=odd)."""
    return encoded


# ---------------------------------------------------------------------------
# stage2 — subset-of-64 selection, plus write footprint / legal offsets
# ---------------------------------------------------------------------------

STAGE2_NAMES: tuple[str, ...] = ("64_128", "32_64", "16_32", "8_16")

# stage2 encodings that write the full 64 elements (no reduction; 8_16 pads
# its 32 taken elements back out to 64 with interleaved zeros).
_FULL_WIDTH_STAGE2: frozenset[int] = frozenset({0, 3})  # 64_128, 8_16


def get_write_width(stage2_encoded: int) -> int:
    """Number of R_ACC elements written for encoded stage2 (64 or 32)."""
    return 64 if stage2_encoded in _FULL_WIDTH_STAGE2 else 32


def get_legal_offsets(stage2_encoded: int) -> tuple[int, ...]:
    """Legal write-offset element values (into R_ACC) for encoded stage2."""
    return (0, 64) if get_write_width(stage2_encoded) == 64 else (0, 32, 64, 96)


def invert_allowed(stage2_encoded: int) -> bool:
    """stage2=64_128 (encoded 0) has no invert option."""
    return stage2_encoded != 0


# ---------------------------------------------------------------------------
# invert — flips which half/group of stage1's output stage2 keeps
# ---------------------------------------------------------------------------

INVERT_NAMES: tuple[str, ...] = ("off", "invert")


def get_invert_bit(encoded: int) -> bool:
    """True if encoded invert selects the inverted (odd-numbered) subset."""
    return bool(encoded)


# ---------------------------------------------------------------------------
# write offset — start index in R_ACC (element units), from a live LR value
# ---------------------------------------------------------------------------


def get_write_offset(lr_value: int) -> int:
    """Start index (element units) in R_ACC: (lr_value % 4) * 32."""
    return (lr_value % 4) * 32
