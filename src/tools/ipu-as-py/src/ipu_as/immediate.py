import lark

import ipu_as.ipu_token as ipu_token
from ipu_common.incr_mod_pow2_k import (
    LR_MOD_POW2_K_FIELD_BITS,
    LR_MOD_POW2_K_MAX,
    LR_MOD_POW2_K_MIN,
)
from ipu_common.mult_mask_offset import (
    MULT_MASK_OFFSET_FIELD_BITS,
    MULT_MASK_SLOT_COUNT,
)
from ipu_common.acc_stride_enums import (
    ELEMENTS_IN_ROW_NAMES,
    HORIZONTAL_STRIDE_NAMES,
    VERTICAL_STRIDE_NAMES,
)
from ipu_common.acc_agg_enums import AGG_MODE_NAMES, POST_FN_NAMES


class LrImmediateType(ipu_token.NumberToken):
    @classmethod
    def bits(cls) -> int:
        return 16


class LrModPow2KImmediate(ipu_token.IpuToken):
    """Semantic k ∈ [LR_MOD_POW2_K_MIN, LR_MOD_POW2_K_MAX]; encoded as (k−1) in LR_MOD_POW2_K_FIELD_BITS bits."""

    @classmethod
    def bits(cls) -> int:
        return LR_MOD_POW2_K_FIELD_BITS

    @classmethod
    def default(cls) -> "ipu_token.IpuToken":
        return cls(
            ipu_token.AnnotatedToken(
                lark.Token("NUMBER", str(LR_MOD_POW2_K_MIN)), 0
            )
        )

    def __init__(self, token: ipu_token.AnnotatedToken):
        super().__init__(token)
        try:
            self.int = int(token.token.value, 0)
        except ValueError:
            self._raise_error(f"Value {self.token.value} is not a valid integer")
        if not (LR_MOD_POW2_K_MIN <= self.int <= LR_MOD_POW2_K_MAX):
            self._raise_error(
                f"Value {self.int} out of range [{LR_MOD_POW2_K_MIN}, {LR_MOD_POW2_K_MAX}] "
                "for incr_mod_pow2 k operand"
            )

    def encode(self) -> int:
        return self.int - LR_MOD_POW2_K_MIN

    @classmethod
    def decode(cls, value: int) -> str:
        return str(value + LR_MOD_POW2_K_MIN)


class BreakImmediateType(ipu_token.NumberToken):
    """Immediate value for break.ifeq condition comparison."""
    @classmethod
    def bits(cls) -> int:
        return 16


class MultMaskOffsetImmediate(ipu_token.IpuToken):
    """Select one of eight 128-bit mask slots in ``r_mask`` (values 0 .. 7)."""

    @classmethod
    def bits(cls) -> int:
        return MULT_MASK_OFFSET_FIELD_BITS

    @classmethod
    def default(cls) -> "ipu_token.IpuToken":
        return cls(
            ipu_token.AnnotatedToken(
                lark.Token("NUMBER", "0"),
                0,
            )
        )

    def __init__(self, token: ipu_token.AnnotatedToken):
        super().__init__(token)
        try:
            self.int = int(token.token.value, 0)
        except ValueError:
            self._raise_error(f"Value {self.token.value} is not a valid integer")
        if not (0 <= self.int < MULT_MASK_SLOT_COUNT):
            self._raise_error(
                f"Value {self.int} out of range [0, {MULT_MASK_SLOT_COUNT - 1}] "
                "for mult mask slot selector"
            )

    def encode(self) -> int:
        return self.int

    @classmethod
    def decode(cls, value: int) -> str:
        return str(value)


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
