"""Generate C headers and SystemVerilog packages from the instruction format.

Single source of truth: ``instruction_spec.py`` (via assembler token/union layout).
Output is produced at build or CLI time and is not checked into the repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from ipu_as import compound_inst, ipu_token, utils
from ipu_as import inst as inst_module

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _sanitize_enum_member(name: str) -> str:
    return name.upper().replace(".", "_").replace("-", "_")


def _enum_descriptors() -> list[dict[str, Any]]:
    """EnumToken subclasses (immediates, register indices, etc.; not slot opcodes)."""
    result: list[dict[str, Any]] = []
    for enum_name, values in ipu_token.EnumToken.get_all_enum_descriptors().items():
        # Slot opcodes are emitted per Inst struct; skip duplicate InstOpcode enums.
        if enum_name.endswith("_inst_opcode"):
            continue
        width = max(1, (len(values) - 1).bit_length()) if len(values) > 1 else 1
        result.append(
            {
                "name": enum_name,
                "c_type": f"{enum_name}_t",
                "sv_type": f"{enum_name}_e",
                "width": width,
                "members": [
                    {"value": idx, "name": _sanitize_enum_member(name)}
                    for idx, name in values
                ],
            }
        )
    return result


def _slot_field_name(inst_cls: type, token_index: int, token_cls: type) -> str:
    base = utils.camel_case_to_snake_case(inst_cls.__name__)
    token_part = utils.camel_case_to_snake_case(token_cls.__name__)
    return f"{base}_token_{token_index}_{token_part}"


def _slot_descriptors() -> list[dict[str, Any]]:
    """Per-slot packed struct layouts (one struct per Inst subclass)."""
    slots: list[dict[str, Any]] = []
    for inst_cls in sorted(
        inst_module.Inst.__subclasses__(), key=lambda c: c.__name__
    ):
        slot_type = inst_cls._slot_type_name()
        opcode_cls = inst_cls.opcode_type()
        opcode_names = opcode_cls.enum_array()
        token_types = inst_cls.all_tokens()
        fields: list[dict[str, Any]] = []
        # MSB-first field order matches packed struct convention (opcode at MSB).
        for idx, tok_cls in enumerate(reversed(token_types)):
            fields.append(
                {
                    "name": _slot_field_name(inst_cls, idx, tok_cls),
                    "bits": tok_cls.bits(),
                    "token_class": tok_cls.__name__,
                }
            )
        struct_base = utils.camel_case_to_snake_case(inst_cls.__name__)
        slots.append(
            {
                "slot": slot_type,
                "struct_name": f"{struct_base}_t",
                "sv_struct": f"{struct_base}_t",
                "width": inst_cls.bits(),
                "opcode_enum": f"{struct_base}_opcode_e",
                "opcode_prefix": struct_base.upper(),
                "opcode_width": opcode_cls.bits(),
                "opcodes": [
                    {
                        "value": idx,
                        "name": _sanitize_enum_member(name),
                    }
                    for idx, name in enumerate(opcode_names)
                ],
                "fields": fields,
            }
        )
    return slots


def _compound_descriptor() -> dict[str, Any]:
    """Flat compound-instruction struct (full VLIW word)."""
    fields = []
    for name, bits in compound_inst.CompoundInst.get_fields():
        fields.append({"name": name, "bits": bits})
    return {
        "struct_name": "ipu_compound_inst_t",
        "sv_struct": "ipu_compound_inst_t",
        "width": compound_inst.CompoundInst.bits(),
        "fields": fields,
    }


def build_codegen_context() -> dict[str, Any]:
    """Build the Jinja render context from live assembler metadata."""
    return {
        "enums": _enum_descriptors(),
        "slots": _slot_descriptors(),
        "compound": _compound_descriptor(),
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
