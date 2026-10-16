"""IPU Register definitions — imported from ipu-isa with backward compatibility.

This module now imports register definitions from ipu-isa, the single source
of truth for all IPU register metadata. It maintains full backward compatibility
by re-exporting all the same names and classes.

Old code uses: from ipu_as.reg import MultStageRegField, LrRegField, etc.
New code uses: from ipu_isa.registers import create_regfile_schema, etc.
"""

import ipu_as.ipu_token as ipu_token
from ipu_isa.operand_types import operand_type
from ipu_isa.registers import REGISTER_DEFINITIONS, create_assembler_reg_enums


def create_assembler_reg_classes() -> dict[str, type]:
    """Generate an ``EnumToken`` subclass per register encoding class, from
    ``ipu_isa.registers.create_assembler_reg_enums()``."""
    enums = create_assembler_reg_enums()
    generated_classes = {}

    for class_name, enum_values in enums.items():
        # Create a new EnumToken subclass dynamically
        def make_class(name, values):
            class GeneratedRegField(ipu_token.EnumToken):
                @classmethod
                def enum_array(cls):
                    return values

            GeneratedRegField.__name__ = name
            GeneratedRegField.__qualname__ = name
            return GeneratedRegField

        generated_classes[class_name] = make_class(class_name, enum_values)

    return generated_classes


# ============================================================================
# Backward Compatibility: Re-export all original names
# ============================================================================

# Constants (for reference/reference)
IPU_MULT_STAGE_REG_R_NUM = REGISTER_DEFINITIONS["r"]["count"]
IPU_LR_REG_NUM = REGISTER_DEFINITIONS["lr"]["count"]
IPU_CR_REG_NUM = REGISTER_DEFINITIONS["cr"]["count"]

# Generate the EnumToken classes from ipu-isa
_generated_classes = create_assembler_reg_classes()

_MultStageBase = _generated_classes.pop("MultStageRegField")


class MultStageRegField(_MultStageBase):
    """Mult-stage field: **2 bits** in the VLIW word; assembly allows only ``r0`` and ``r1``."""

    @classmethod
    def bits(cls) -> int:
        return operand_type("MultStageReg").bits

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


IPU_LRD_REG_NUM = IPU_LR_REG_NUM // 2


class LrdRegField(ipu_token.EnumToken):
    """LRD0, LRD2, LRD4, ..., LRD14: register-pair alias over LR, named after
    the lower register in the pair (LRDn = LR(n+1):LR(n), n even).

    Not a physical register — no storage of its own. ``ADDB``/``ADDBI`` resolve
    an ``LrdIdx`` encoded index ``i`` (0-7) to the real ``LR(2i)``/``LR(2i+1)``
    pair directly.
    """

    @classmethod
    def enum_array(cls) -> list[str]:
        return list(operand_type("LrdIdx").domain.members)


# For documentation and introspection, also expose the enum arrays
_enums = create_assembler_reg_enums()
MULT_STAGE_REG_R_FIELDS = _enums.get("MultStageRegField", [])
LR_REG_FIELDS = _enums.get("LrRegField", [])
CR_REG_FIELDS = _enums.get("CrRegField", [])
LCR_REG_FIELDS = _enums.get("LcrRegField", [])
LRD_REG_FIELDS = LrdRegField.enum_array()

# Clean up internal state
del _generated_classes, _enums

__all__ = [
    # Constants
    "IPU_MULT_STAGE_REG_R_NUM",
    "IPU_LR_REG_NUM",
    "IPU_CR_REG_NUM",
    "IPU_LRD_REG_NUM",
    # Classes
    "MultStageRegField",
    "LrRegField",
    "CrRegField",
    "LcrRegField",
    "DstructureCrRegField",
    "LrdRegField",
    # Field lists
    "MULT_STAGE_REG_R_FIELDS",
    "LR_REG_FIELDS",
    "CR_REG_FIELDS",
    "LCR_REG_FIELDS",
    "LRD_REG_FIELDS",
]
