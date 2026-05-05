import lark

import ipu_as.ipu_token as ipu_token
from ipu_common.acc_stride_enums import (
    ELEMENTS_IN_ROW_NAMES,
    HORIZONTAL_STRIDE_NAMES,
    VERTICAL_STRIDE_NAMES,
)
from ipu_common.acc_agg_enums import AGG_MODE_NAMES, POST_FN_NAMES

# incr_mod_pow2 k operand — matches ISA (Issue #47)
LR_MOD_POW2_K_MIN = 1
LR_MOD_POW2_K_MAX = 9


class LrImmediateType(ipu_token.NumberToken):
    @classmethod
    def bits(cls) -> int:
        return 16


class LrModPow2KImmediate(LrImmediateType):
    """Immediate k for incr_mod_pow2: must be in [LR_MOD_POW2_K_MIN, LR_MOD_POW2_K_MAX] per ISA."""

    @classmethod
    def default(cls) -> "ipu_token.IpuToken":
        return cls(
            ipu_token.AnnotatedToken(
                lark.Token("NUMBER", str(LR_MOD_POW2_K_MIN)), 0
            )
        )

    def __init__(self, token: ipu_token.AnnotatedToken):
        super().__init__(token)
        if not (LR_MOD_POW2_K_MIN <= self.int <= LR_MOD_POW2_K_MAX):
            self._raise_error(
                f"Value {self.int} out of range [{LR_MOD_POW2_K_MIN}, {LR_MOD_POW2_K_MAX}] "
                "for incr_mod_pow2 k operand"
            )


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


# ---------------------------------------------------------------------------
# acc.agg operand enums (instruction-specific)
# Single source of truth: ipu_common.acc_agg_enums
# ---------------------------------------------------------------------------

class AggModeField(ipu_token.EnumToken):
    """Aggregation mode: sum or max."""
    @classmethod
    def enum_array(cls) -> list[str]:
        return list(AGG_MODE_NAMES)


class PostFnField(ipu_token.EnumToken):
    """Post function: value, value_cr, inv, inv_sqrt."""
    @classmethod
    def enum_array(cls) -> list[str]:
        return list(POST_FN_NAMES)
