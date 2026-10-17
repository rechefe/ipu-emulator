"""Operand types to token classes, and the instruction reference in Markdown.

Encoding is not here: ``ipu_as.encode`` places instructions and the ``ipu_isa``
codec owns the bits (E05-02).
"""

from collections.abc import Mapping, Sequence

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
from ipu_isa.instruction_spec import InstructionDoc, syntax
from ipu_isa.operand_types import OPERAND_TYPES, OperandType
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.view import Isa, Operand, isa


def _operand_type_md_link(typ: str) -> str:
    slug = typ.lower().replace("_", "-")
    return f"[`{typ}`](operand-types.md#{slug})"


def _md_table_cell(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


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


# ===========================================================================
# Instruction reference (Markdown), for the generated docs
# ===========================================================================


def render_slot_docs(slot: str, view: Isa | None = None) -> str:
    """The reference section of one slot, in Markdown.

    The heading and intro come from the slot registry (``SlotInfo.doc_heading``,
    ``doc_intro``), then one entry per documented instruction.

    Args:
        slot: The slot name, e.g. ``"lr"``.
        view: The ISA; default ``isa()``.

    Returns:
        The section text.

    Raises:
        ValueError: ``slot`` is no slot.
    """
    view = isa() if view is None else view
    info = view.slots[Slot(slot)]
    heading, intro = info.doc_heading, info.doc_intro
    lines = [f"## {heading}", ""]
    if info.tag is Tag.EMULATOR:
        lines.append(
            "> **Emulator-only slot** — not implemented in real IPU hardware: not in the "
            "hardware word or the SV package."
        )
        lines.append("")
    if intro.strip():
        lines.append(intro.strip())
        lines.append("")

    for instruction in view.instructions(info.slot):
        if instruction.mnemonic == "NOP":
            continue  # documented once globally in the Slots section
        if instruction.doc is None:
            continue
        lines.extend(
            render_opcode_doc(
                instruction.mnemonic, instruction.doc, instruction.operands, instruction.tag
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip()


#: How the instruction reference describes each tag (E03-23).
TAG_TEXT = {
    Tag.HARDWARE: "implemented by the IPU",
    Tag.EMULATOR: "emulator only; not in the hardware word or the SV package",
    Tag.ASSEMBLER: "expanded by the assembler; no opcode",
}


def render_opcode_doc(
    opcode: str,
    doc: InstructionDoc,
    spec_operands: Sequence[Operand],
    tag: Tag | None = None,
) -> list[str]:
    """Markdown documentation of one opcode or pseudo-instruction.

    Args:
        opcode: The mnemonic, as written in assembly.
        doc: Its ``InstructionDoc``.
        spec_operands: Its operand dicts from the instruction spec, in order.
        tag: Its tag, shown under the syntax; omitted when ``None``.

    Returns:
        The Markdown lines: heading, syntax, tag, operands, and the doc's prose.
    """
    lines = [f"### `{opcode}` — {doc.title}", ""]
    names = [operand.name for operand in spec_operands]
    lines.append(f"**Syntax:** `{syntax(opcode, names)}`")
    lines.append("")
    if tag is not None:
        lines.append(f"**Tag:** `{tag.name}` ({TAG_TEXT[tag]})")
        lines.append("")

    if spec_operands:
        lines.append(
            "**Operands:** *(the **Type** column links to the "
            "[operand type reference](operand-types.md))*"
        )
        lines.append("")
        lines.append("| Name | Type | Details |")
        lines.append("|------|------|---------|")
        for i, sop in enumerate(spec_operands):
            name = sop.name
            typ = sop.type.name
            link = _operand_type_md_link(typ)
            detail = doc.operands[i] if i < len(doc.operands) else "—"
            lines.append(f"| `{name}` | {link} | {_md_table_cell(detail)} |")
        lines.append("")
    elif doc.operands:
        lines.append("**Operands:**")
        lines.extend([f"- {operand}" for operand in doc.operands])
        lines.append("")

    lines.append("**General description:**")
    lines.append(doc.summary)
    lines.append("")

    if doc.notes:
        lines.append(f"> {doc.notes}")
        lines.append("")

    if doc.operation:
        lines.append("**Pseudo code:**")
        lines.append(f"`{doc.operation}`")
        lines.append("")

    if doc.example:
        lines.append("**Example of usage:**")
        lines.append("```asm")
        lines.append(doc.example)
        lines.append("```")

    return lines
