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


CrRegField = _CrRegFieldBase
LcrRegField = _LcrRegFieldBase


class DstructureCrRegField(_CrRegFieldBase):
    """CR0–CR15 selector for an instruction's dstructure (valid-element mask) operand.

    This operand is mandatory — every AGG.*/AAQ/ACTIVATE instruction must name
    its dstructure CR explicitly (e.g. ``AAQ cr15;;``); CR15 is the conventional
    choice but any CR0–CR15 works.
    """


IPU_LRC_REG_NUM = 8


class LrcRegField(ipu_token.EnumToken):
    """LRC0, LRC2, LRC4, ..., LRC14: register-pair alias over LR, named after
    the lower register in the pair (LRCn = LR(n+1):LR(n), n even).

    Not a physical register — no storage of its own. ``ADDB``/``ADDBI`` resolve
    an ``LrcIdx`` encoded index ``i`` (0-7) to the real ``LR(2i)``/``LR(2i+1)``
    pair directly.
    """

    @classmethod
    def enum_array(cls) -> list[str]:
        return [f"lrc{2 * i}" for i in range(IPU_LRC_REG_NUM)]


# For documentation and introspection, also expose the enum arrays
_enums = create_assembler_reg_enums()
MULT_STAGE_REG_R_FIELDS = _enums.get("MultStageRegField", [])
LR_REG_FIELDS = _enums.get("LrRegField", [])
CR_REG_FIELDS = _enums.get("CrRegField", [])
LCR_REG_FIELDS = _enums.get("LcrRegField", [])
LRC_REG_FIELDS = LrcRegField.enum_array()

# Clean up internal state
del _generated_classes, _enums

__all__ = [
    # Constants
    "IPU_MULT_STAGE_REG_R_NUM",
    "IPU_LR_REG_NUM",
    "IPU_CR_REG_NUM",
    "IPU_LRC_REG_NUM",
    # Classes
    "MultStageRegField",
    "LrRegField",
    "CrRegField",
    "LcrRegField",
    "DstructureCrRegField",
    "LrcRegField",
    # Field lists
    "MULT_STAGE_REG_R_FIELDS",
    "LR_REG_FIELDS",
    "CR_REG_FIELDS",
    "LCR_REG_FIELDS",
    "LRC_REG_FIELDS",
]
