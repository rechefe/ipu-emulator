"""Operand types to token classes.

The instruction reference is rendered by ``ipu_gen.docs_pages`` (E09-14).
Encoding is not here: ``ipu_as.encode`` places instructions and the ``ipu_isa``
codec owns the bits (E05-02).
"""

from collections.abc import Mapping

# isort: off
# Import order defines EnumToken subclass registration order, which fixes the
# enum order in generated outputs (ipu_instr_pkg.sv, disassembly): opcodes
# first, then registers, then immediates.
import ipu_as.opcodes as opcodes  # noqa: F401 - registers the opcode enums first
import ipu_as.ipu_token as ipu_token
import ipu_as.reg as reg
import ipu_as.immediate as immediate
import ipu_as.utils as utils

# isort: on
from ipu_isa.operand_types import OPERAND_TYPES, OperandType

# ===========================================================================
# Operand Type Resolution
# ===========================================================================
# Maps string type names from instruction_spec → actual token classes.
# This is the ONLY place where type names are resolved to Python classes.
# ===========================================================================

OPERAND_TYPE_MAP: dict[str, type[ipu_token.IpuToken]] = {
    "MultStageReg": reg.MultStageRegField,
    "LrIdx": reg.LrRegField,
    "CrIdx": reg.CrRegField,
    "DstructureCrIdx": reg.DstructureCrRegField,
    "LcrIdx": reg.LcrRegField,
    "LrdIdx": reg.LrdRegField,
    "LrIncDecImmediate": immediate.LrIncDecImmediate,
    "AddbiImmediate": immediate.AddbiImmediate,
    "ElementsInRow": immediate.ElementsInRowField,
    "HorizontalStride": immediate.HorizontalStrideField,
    "VerticalStride": immediate.VerticalStrideField,
    "LrModPow2KImmediate": immediate.LrModPow2KImmediate,
    "MultMaskOffsetImmediate": immediate.MultMaskOffsetImmediate,
    "LrOrReshapeMaskImmediate": immediate.LrOrReshapeMaskImmediate,
    "ActivationFn": immediate.ActivationFnField,
    "BreakImmediate": immediate.BreakImmediateType,
    "Label": ipu_token.LabelToken,
}


def check_operand_type_map(
    type_map: Mapping[str, type[ipu_token.IpuToken]],
    operand_types: Mapping[str, OperandType] = OPERAND_TYPES,
) -> None:
    """The token classes agree with the operand-type registry.

    Every registry type has a token class; the class is the one the registry
    names, yields the registry's field suffix, and has the registry's width and
    value range; an enum token's members are the registry domain, in order, and
    its SV typedef the registry's. Raises ``ValueError`` naming the type.
    """
    if set(type_map) != set(operand_types):
        missing = sorted(set(operand_types) - set(type_map))
        extra = sorted(set(type_map) - set(operand_types))
        raise ValueError(
            f"OPERAND_TYPE_MAP disagrees with the operand-type registry: "
            f"missing {missing}, not in the registry {extra}"
        )
    for name, token_class in type_map.items():
        _check_token_class(name, token_class, operand_types[name])


def _check_token_class(
    name: str, token_class: type[ipu_token.IpuToken], record: OperandType
) -> None:
    """One token class against its registry record (``check_operand_type_map``)."""
    if token_class.__name__ != record.token_class:
        raise ValueError(
            f"OPERAND_TYPE_MAP[{name!r}] is {token_class.__name__}, but the "
            f"operand-type registry names {record.token_class}"
        )
    # Decoded field names and SV typedefs are derived from the token class
    # name today; they must be the wire names the registry writes out.
    suffix = utils.camel_case_to_snake_case(token_class.__name__)
    if suffix != record.field_suffix:
        raise ValueError(
            f"{name}: token class {token_class.__name__} yields field suffix "
            f"{suffix!r}, but the registry says {record.field_suffix!r}"
        )
    if record.bits is not None and token_class.bits() != record.bits:
        raise ValueError(
            f"{name}: {token_class.__name__} is {token_class.bits()} bits, "
            f"but the registry says {record.bits}"
        )
    if record.domain.value_range is not None and hasattr(token_class, "value_range"):
        if tuple(token_class.value_range()) != record.domain.value_range:
            raise ValueError(
                f"{name}: {token_class.__name__} accepts {token_class.value_range()}, "
                f"but the registry says {record.domain.value_range}"
            )
    if issubclass(token_class, ipu_token.EnumToken):
        members = tuple(token_class.enum_array())
        if members != record.domain.members:
            raise ValueError(
                f"{token_class.__name__} members {members} differ from the "
                f"operand-type registry domain of {name}: {record.domain.members}"
            )
        if record.sv_typedef != f"{suffix}_t":
            raise ValueError(
                f"{name}: SV typedef {record.sv_typedef!r} is not the enum "
                f"descriptor name {suffix + '_t'!r} the package declares"
            )


check_operand_type_map(OPERAND_TYPE_MAP)
