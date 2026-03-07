import ipu_as.ipu_token as ipu_token
from ipu_common.acc_stride_enums import (
    ELEMENTS_IN_ROW_NAMES,
    HORIZONTAL_STRIDE_NAMES,
    VERTICAL_STRIDE_NAMES,
)


class LrImmediateType(ipu_token.NumberToken):
    @classmethod
    def bits(cls) -> int:
        return 16


class BreakImmediateType(ipu_token.NumberToken):
    """Immediate value for break.ifeq condition comparison."""
    @classmethod
    def bits(cls) -> int:
        return 16


# ---------------------------------------------------------------------------
# acc.stride operand enums (instruction-specific, not registers)
# Single source of truth: ipu_common.acc_stride_enums
# ---------------------------------------------------------------------------

class ElementsInRowField(ipu_token.EnumToken):
    """Elements per row: 8, 16, 32, or 64."""
    @classmethod
    def enum_array(cls) -> list[str]:
        return list(ELEMENTS_IN_ROW_NAMES)


class HorizontalStrideField(ipu_token.EnumToken):
    """Horizontal stride: enabled(1), inverted(2), expand(3). Bits 0..2."""
    @classmethod
    def enum_array(cls) -> list[str]:
        return list(HORIZONTAL_STRIDE_NAMES)


class VerticalStrideField(ipu_token.EnumToken):
    """Vertical stride: enabled(1), inverted(2). Bits 0..1."""
    @classmethod
    def enum_array(cls) -> list[str]:
        return list(VERTICAL_STRIDE_NAMES)
