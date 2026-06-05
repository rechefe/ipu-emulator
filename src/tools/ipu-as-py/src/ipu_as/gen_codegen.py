"""Generate C headers and SystemVerilog packages from the instruction format.

Mirrors the historical C-header generator: ``EnumToken`` descriptors plus
``CompoundInst.get_fields()`` for the flat wire-level word.  The SystemVerilog
package additionally emits per-slot union-layout structs and per-instruction
``union packed`` views derived from ``SLOT_UNIONS`` in ``instruction_spec``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from ipu_as import compound_inst, ipu_token, utils
from ipu_common.instruction_spec import INSTRUCTION_SPEC, SLOT_COUNT, SLOT_UNIONS
from ipu_common.union_layout import get_operand_type_bits

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Operand type string → generated SystemVerilog enum typedef (when applicable).
_OPERAND_TYPE_TO_SV_TYPEDEF: dict[str, str] = {
    "MultStageReg": "mult_stage_reg_field_t",
    "LrIdx": "lr_reg_field_t",
    "CrIdx": "cr_reg_field_t",
    "LcrIdx": "lcr_reg_field_t",
    "AddSubSrcB": "add_sub_src_b_field_t",
    "AaqRegIdx": "aaq_reg_field_t",
    "ElementsInRow": "elements_in_row_field_t",
    "HorizontalStride": "horizontal_stride_field_t",
    "VerticalStride": "vertical_stride_field_t",
    "AggMode": "agg_mode_field_t",
    "PostFn": "post_fn_field_t",
    "ActivationFn": "activation_fn_field_t",
    "FullXmemRow": "full_xmem_row_field_t",
}


def _sv_sized_literal(width: int, value: int) -> str:
    """SystemVerilog sized integer literal, e.g. width=3 value=5 → ``3'd5``."""
    return f"{width}'d{value}"

# Slot name → opcode EnumToken descriptor key and struct basename.
_SV_RESERVED_STRUCT_NAMES = frozenset({
    "break",
    "continue",
    "return",
    "module",
    "endmodule",
    "begin",
    "end",
    "case",
    "default",
    "function",
    "task",
})

_SLOT_META: dict[str, tuple[str, str]] = {
    "cond": ("cond_inst_opcode", "cond_slot"),
    "lr": ("lr_inst_opcode", "lr_slot"),
    "xmem": ("xmem_inst_opcode", "xmem_slot"),
    "mult": ("mult_inst_opcode", "mult_slot"),
    "acc": ("acc_inst_opcode", "acc_slot"),
    "aaq": ("aaq_inst_opcode", "aaq_slot"),
    "break": ("break_inst_opcode", "break_slot"),
}


def _sanitize_enum_member(name: str) -> str:
    return name.upper().replace(".", "_").replace("-", "_")


def _sv_logic_type(canonical_type: str, bits: int) -> str:
    typedef_name = _OPERAND_TYPE_TO_SV_TYPEDEF.get(canonical_type)
    if typedef_name is not None:
        return typedef_name
    return f"logic [{bits - 1}:0]"


def _canonical_field_name(canonical_type: str, field_index: int) -> str:
    base = utils.camel_case_to_snake_case(canonical_type)
    return f"{base}_{field_index}"


def _instruction_struct_name(inst_name: str) -> str:
    base = _sanitize_enum_member(inst_name).lower()
    if base in _SV_RESERVED_STRUCT_NAMES:
        return f"{base}_inst"
    return base


def _slot_union_descriptors() -> list[dict[str, Any]]:
    """Per-slot union layout structs and per-instruction union members."""
    type_bits = get_operand_type_bits()
    slots: list[dict[str, Any]] = []

    for slot_name, slot_union in SLOT_UNIONS.items():
        opcode_key, struct_base = _SLOT_META[slot_name]
        opcode_enum = f"{opcode_key}_t"
        fields: list[dict[str, Any]] = []
        for uf in slot_union.fields:
            fields.append(
                {
                    "name": _canonical_field_name(uf.canonical_type, uf.index),
                    "bits": uf.bits,
                    "canonical_type": uf.canonical_type,
                    "sv_type": _sv_logic_type(uf.canonical_type, uf.bits),
                }
            )

        instructions: list[dict[str, Any]] = []
        for inst_name, inst_def in INSTRUCTION_SPEC[slot_name].items():
            operands: list[dict[str, Any]] = []
            bindings = slot_union.opcode_bindings.get(inst_name, [])
            operand_types = {op["name"]: op["type"] for op in inst_def["operands"]}
            for _field_idx, operand_name in bindings:
                actual_type = operand_types[operand_name]
                op_bits = type_bits[actual_type]
                operands.append(
                    {
                        "name": operand_name,
                        "sv_type": _sv_logic_type(actual_type, op_bits),
                    }
                )
            instructions.append(
                {
                    "name": inst_name,
                    "sv_struct": _instruction_struct_name(inst_name),
                    "operands": operands,
                }
            )

        opcode_names = list(INSTRUCTION_SPEC[slot_name].keys())
        slot_width = slot_union.opcode_bits + sum(f["bits"] for f in fields)
        slots.append(
            {
                "slot": slot_name,
                "opcode_enum": opcode_enum,
                "opcode_width": slot_union.opcode_bits,
                "opcode_prefix": struct_base.upper(),
                "struct_name": f"{struct_base}_t",
                "union_name": f"{struct_base}_u",
                "width": slot_width,
                "fields": fields,
                "instructions": instructions,
                "opcodes": [
                    {
                        "value": idx,
                        "name": _sanitize_enum_member(name),
                        "sized_value": _sv_sized_literal(
                            slot_union.opcode_bits, idx
                        ),
                    }
                    for idx, name in enumerate(opcode_names)
                ],
            }
        )

    return slots


def _compound_members() -> list[dict[str, Any]]:
    """Nested compound struct members in MSB → LSB order (matches encode layout)."""
    members: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}

    for inst_cls in compound_inst.CompoundInst.instruction_types():
        slot = inst_cls._slot_type_name()
        _, struct_base = _SLOT_META[slot]
        sv_type = f"{struct_base}_t"

        count = type_counts.get(slot, 0)
        type_counts[slot] = count + 1
        if SLOT_COUNT[slot] > 1:
            member_name = f"{struct_base}_{count}"
        else:
            member_name = struct_base

        members.append(
            {
                "name": member_name,
                "sv_type": sv_type,
                "slot": slot,
            }
        )

    return members


def _enum_descriptors_for_templates() -> list[dict[str, Any]]:
    """EnumToken descriptors with precomputed bit-width for SV typedefs."""
    result: list[dict[str, Any]] = []
    for enum_name, members in ipu_token.EnumToken.get_all_enum_descriptors().items():
        n = len(members)
        width = max(1, (n - 1).bit_length()) if n > 1 else 1
        result.append(
            {
                "name": enum_name,
                "c_type": f"{enum_name}_t",
                "sv_type": f"{enum_name}_t",
                "width": width,
                "members": [
                    {
                        "value": value,
                        "name": name,
                        "sized_value": _sv_sized_literal(width, value),
                    }
                    for value, name in members
                ],
            }
        )
    return result


def build_codegen_context() -> dict[str, Any]:
    """Build the Jinja render context from live assembler metadata."""
    enum_list = _enum_descriptors_for_templates()
    return {
        "enums": {e["name"]: [(m["value"], m["name"]) for m in e["members"]] for e in enum_list},
        "enum_types": enum_list,
        "inst_bit_fields": compound_inst.CompoundInst.get_fields(),
        "slots": _slot_union_descriptors(),
        "compound_members": _compound_members(),
        "compound_width": compound_inst.CompoundInst.bits(),
    }


def _template_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(template_name: str, context: dict[str, Any] | None = None) -> str:
    """Render a named template with the instruction-format context."""
    ctx = context if context is not None else build_codegen_context()
    return _template_env().get_template(template_name).render(**ctx)


def write_generated_file(template_name: str, output_path: str | Path) -> None:
    """Render *template_name* and write to *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_template(template_name), encoding="utf-8")


def generate_c_header(output_path: str | Path) -> None:
    """Generate a C header describing the compound instruction bit layout."""
    write_generated_file("ipu_inst.h.j2", output_path)


def generate_sv_package(output_path: str | Path) -> None:
    """Generate a SystemVerilog package with instruction-format structs and enums."""
    write_generated_file("ipu_instr_pkg.sv.j2", output_path)
