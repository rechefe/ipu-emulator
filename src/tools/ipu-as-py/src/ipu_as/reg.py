"""IPU Register definitions — imported from ipu-common with backward compatibility.

This module now imports register definitions from ipu-common, the single source
of truth for all IPU register metadata. It maintains full backward compatibility
by re-exporting all the same names and classes.

Old code uses: from ipu_as.reg import MultStageRegField, LrRegField, etc.
New code uses: from ipu_common.registers import create_regfile_schema, etc.
"""

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

# Re-export generated classes with their original names
MultStageRegField = _generated_classes.get("MultStageRegField")
LrRegField = _generated_classes.get("LrRegField")
CrRegField = _generated_classes.get("CrRegField")
LcrRegField = _generated_classes.get("LcrRegField")
AaqRegField = _generated_classes.get("AaqRegField")


class MultStageRegR01Field(MultStageRegField):
    """Mult-stage operand for ``mult.ee``: same 2-bit encoding as ``MultStageRegField`` but ``mem_bypass`` is rejected."""

    def __init__(self, token: ipu_token.AnnotatedToken):
        if token.token.value.lower() == "mem_bypass":
            loc = f"Line {token.token.line}, Column {token.token.column}"
            raise ValueError(
                "Invalid token value - mem_bypass in token MultStageRegR01Field\n"
                f"In {loc}\n"
                "`mem_bypass` is not valid for `mult.ee`; use `r0` or `r1` "
                "(load with `ldr_mult_reg` in the same compound if needed)."
            )
        super().__init__(token)

# For documentation and introspection, also expose the enum arrays
_enums = create_assembler_reg_enums()
MULT_STAGE_REG_R_FIELDS = _enums.get("MultStageRegField", [])
LR_REG_FIELDS = _enums.get("LrRegField", [])
CR_REG_FIELDS = _enums.get("CrRegField", [])
LCR_REG_FIELDS = _enums.get("LcrRegField", [])
AAQ_REG_FIELDS = _enums.get("AaqRegField", [])

# Clean up internal state
del _generated_classes, _enums

__all__ = [
    # Constants
    "IPU_MULT_STAGE_REG_R_NUM",
    "IPU_LR_REG_NUM",
    "IPU_CR_REG_NUM",
    # Classes
    "MultStageRegField",
    "MultStageRegR01Field",
    "LrRegField",
    "CrRegField",
    "LcrRegField",
    "AaqRegField",
    # Field lists
    "MULT_STAGE_REG_R_FIELDS",
    "LR_REG_FIELDS",
    "CR_REG_FIELDS",
    "LCR_REG_FIELDS",
    "AAQ_REG_FIELDS",
]
