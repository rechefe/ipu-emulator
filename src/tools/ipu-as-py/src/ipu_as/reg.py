"""IPU Register definitions — imported from ipu-common with backward compatibility.

This module now imports register definitions from ipu-common, the single source
of truth for all IPU register metadata. It maintains full backward compatibility
by re-exporting all the same names and classes.

Old code uses: from ipu_as.reg import MultStageRegField, LrRegField, etc.
New code uses: from ipu_common.registers import create_regfile_schema, etc.
"""

import lark
import ipu_as.ipu_token as ipu_token
from ipu_common.registers import create_assembler_reg_classes, create_assembler_reg_enums

# ============================================================================
# Backward Compatibility: Re-export all original names
# ============================================================================

# Constants (for reference/reference)
IPU_MULT_STAGE_REG_R_NUM = 2
IPU_LR_REG_NUM = 16
IPU_CR_REG_NUM = 16

# Generate the EnumToken classes from ipu-common
_generated_classes = create_assembler_reg_classes(ipu_token)

_MultStageBase = _generated_classes.pop("MultStageRegField")


class MultStageRegField(_MultStageBase):
    """Mult-stage field: **2 bits** in the VLIW word; assembly allows only ``r0`` and ``r1``."""

    @classmethod
    def bits(cls) -> int:
        return 2

    @classmethod
    def decode(cls, value: int) -> str:
        opts = cls.enum_array()
        if 0 <= value < len(opts):
            return opts[value]
        return f"<illegal_mult_stage_{value}>"


# Re-export generated classes with their original names
LrRegField = _generated_classes.get("LrRegField")
_CrRegFieldBase = _generated_classes.get("CrRegField")
_LcrRegFieldBase = _generated_classes.get("LcrRegField")


def _reject_cr15(token: "ipu_token.AnnotatedToken", cls_name: str) -> None:
    if token.token.value.lower() == "cr15":
        raise ValueError(
            f"CR15 is reserved for dstructure configuration and cannot be used as an ISA operand.\n"
            f"In Line {token.token.line}, Column {token.token.column}"
        )


class CrRegField(_CrRegFieldBase):
    pass


class LcrRegField(_LcrRegFieldBase):
    def __init__(self, token: ipu_token.AnnotatedToken):
        _reject_cr15(token, "LcrRegField")
        super().__init__(token)


class CrDstructureIdxField(_CrRegFieldBase):
    """CR register index for dstructure operands: accepts CR0–CR15 (including CR15), defaults to CR15."""

    @classmethod
    def default(cls) -> "ipu_token.IpuToken":
        return cls(ipu_token.AnnotatedToken(lark.Token("TOKEN", "cr15"), 0))

# For documentation and introspection, also expose the enum arrays
_enums = create_assembler_reg_enums()
MULT_STAGE_REG_R_FIELDS = _enums.get("MultStageRegField", [])
LR_REG_FIELDS = _enums.get("LrRegField", [])
CR_REG_FIELDS = _enums.get("CrRegField", [])
LCR_REG_FIELDS = _enums.get("LcrRegField", [])

# Clean up internal state
del _generated_classes, _enums

__all__ = [
    # Constants
    "IPU_MULT_STAGE_REG_R_NUM",
    "IPU_LR_REG_NUM",
    "IPU_CR_REG_NUM",
    # Classes
    "MultStageRegField",
    "LrRegField",
    "CrRegField",
    "LcrRegField",
    "CrDstructureIdxField",
    # Field lists
    "MULT_STAGE_REG_R_FIELDS",
    "LR_REG_FIELDS",
    "CR_REG_FIELDS",
    "LCR_REG_FIELDS",
]
