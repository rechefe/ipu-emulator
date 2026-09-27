"""Generate SystemVerilog packages from the instruction format.

Emits ``EnumToken`` descriptors, per-slot structs (opcode + operand union) and
per-instruction ``union packed`` views derived from ``SLOT_UNIONS`` in
``instruction_spec``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import jinja2

from ipu_as import compound_inst, ipu_token, utils
from ipu_as.inst import OPERAND_TYPE_MAP
from ipu_common.instruction_spec import (
    INSTRUCTION_SPEC,
    SLOT_COUNT,
    SLOT_UNIONS,
    VALID_OPERAND_TYPES,
)
from ipu_common.union_layout import get_operand_type_bits

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Operand type string → generated SystemVerilog enum typedef. Every operand type
# backed by an EnumToken must appear here (enforced by _check_sv_typedef_map).
_OPERAND_TYPE_TO_SV_TYPEDEF: dict[str, str] = {
    "MultStageReg": "mult_stage_reg_field_t",
    "LrIdx": "lr_reg_field_t",
    "CrIdx": "cr_reg_field_t",
    "LcrIdx": "lcr_reg_field_t",
    "LrdIdx": "lrd_reg_field_t",
    "ElementsInRow": "elements_in_row_field_t",
    "HorizontalStride": "horizontal_stride_field_t",
    "VerticalStride": "vertical_stride_field_t",
    "ActivationFn": "activation_fn_field_t",
    "DstructureCrIdx": "dstructure_cr_reg_field_t",
}

# Stamped into the package header; see tools/workspace_status.sh.
UNKNOWN_SOURCE_COMMIT = "unknown"
_SOURCE_COMMIT_STATUS_KEY = "STABLE_GIT_COMMIT"

# IEEE 1800-2017 Annex B reserved keywords. Generated identifiers must avoid them.
_SV_KEYWORDS = frozenset("""
    accept_on alias always always_comb always_ff always_latch and assert assign
    assume automatic before begin bind bins binsof bit break buf bufif0 bufif1
    byte case casex casez cell chandle checker class clocking cmos config const
    constraint context continue cover covergroup coverpoint cross deassign
    default defparam design disable dist do edge else end endcase endchecker
    endclass endclocking endconfig endfunction endgenerate endgroup endinterface
    endmodule endpackage endprimitive endprogram endproperty endspecify
    endsequence endtable endtask enum event eventually expect export extends
    extern final first_match for force foreach forever fork forkjoin function
    generate genvar global highz0 highz1 if iff ifnone ignore_bins illegal_bins
    implements implies import incdir include initial inout input inside instance
    int integer interconnect interface intersect join join_any join_none large
    let liblist library local localparam logic longint macromodule matches
    medium modport module nand negedge nettype new nexttime nmos nor
    noshowcancelled not notif0 notif1 null or output package packed parameter
    pmos posedge primitive priority program property protected pull0 pull1
    pulldown pullup pulsestyle_ondetect pulsestyle_onevent pure rand randc
    randcase randsequence rcmos real realtime ref reg reject_on release repeat
    restrict return rnmos rpmos rtran rtranif0 rtranif1 s_always s_eventually
    s_nexttime s_until s_until_with scalared sequence shortint shortreal
    showcancelled signed small soft solve specify specparam static string strong
    strong0 strong1 struct super supply0 supply1 sync_accept_on sync_reject_on
    table tagged task this throughout time timeprecision timeunit tran tranif0
    tranif1 tri tri0 tri1 triand trior trireg type typedef union unique unique0
    unsigned until until_with untyped use uwire var vectored virtual void wait
    wait_order wand weak weak0 weak1 while wildcard wire with within wor xnor xor
""".split())


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
    "set",
})

_SLOT_META: dict[str, tuple[str, str]] = {
    "cond": ("cond_inst_opcode", "cond_slot"),
    "lr": ("lr_inst_opcode", "lr_slot"),
    "load": ("load_inst_opcode", "load_slot"),
    "store": ("store_inst_opcode", "store_slot"),
    "acc_store": ("acc_store_inst_opcode", "acc_store_slot"),
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
    if base in _SV_RESERVED_STRUCT_NAMES or base in _SV_KEYWORDS:
        return f"{base}_inst"
    return base


def _operand_sv_type(
    op_type: str,
    column_bits: int,
    type_bits: dict[str, int],
    typedef_bits: dict[str, int],
) -> tuple[str, int]:
    """SV type and width of an operand declared at its own width.

    Raises if the operand does not fit its union column, or if its SV type is
    wider than the bits the assembler reserves for it.
    """
    # Derived-width immediates (width 0 in get_operand_type_bits) are defined
    # as filling the union column they were packed into.
    reserved_bits = type_bits[op_type] or column_bits
    if reserved_bits > column_bits:
        raise ValueError(
            f"{op_type} reserves {reserved_bits} bits but its union column is "
            f"{column_bits} bits"
        )
    typedef_name = _OPERAND_TYPE_TO_SV_TYPEDEF.get(op_type)
    if typedef_name is None:
        return f"logic [{reserved_bits - 1}:0]", reserved_bits
    bits = typedef_bits[typedef_name]
    if bits > reserved_bits:
        raise ValueError(
            f"{typedef_name} is {bits} bits but the assembler reserves only "
            f"{reserved_bits} bits for {op_type}"
        )
    return typedef_name, bits


def _padding_field_name(field_index: int) -> str:
    """SV member name for unused bits of a union column (unique per column index)."""
    return f"padding_{field_index}"


def _padding_field(name: str, bits: int) -> dict[str, Any]:
    return {
        "name": name,
        "sv_type": f"logic [{bits - 1}:0]",
        "bits": bits,
        "operand": "padding",
    }


def _instruction_layout_fields(
    slot_union: Any,
    slot_fields: list[dict[str, Any]],
    inst_name: str,
    inst_def: dict,
    type_bits: dict[str, int],
    typedef_bits: dict[str, int],
) -> list[dict[str, Any]]:
    """Operand-area struct members for a per-instruction union member (MSB → LSB).

    The opcode lives outside ``{slot}_slot_u`` in ``{slot}_slot_t`` — it is shared
    across all instructions in the slot.  Each operand is declared at its own
    width and named ``<operand>_<column>``; unused bits of a column become
    ``padding_<column>``, and operand-less instructions use a single ``padding``
    field for the whole payload.
    """
    bindings = {
        field_idx: op_name
        for field_idx, op_name in slot_union.opcode_bindings.get(inst_name, [])
    }
    operand_types = {op["name"]: op["type"] for op in inst_def["operands"]}

    if not bindings:
        operand_width = sum(f["bits"] for f in slot_fields)
        return [_padding_field("padding", operand_width)]

    layout: list[dict[str, Any]] = []
    for field in slot_fields:
        field_idx = field["index"]
        column_bits = field["bits"]
        if field_idx not in bindings:
            layout.append(_padding_field(_padding_field_name(field_idx), column_bits))
            continue
        op_name = bindings[field_idx]
        op_type = operand_types[op_name]
        sv_type, bits = _operand_sv_type(op_type, column_bits, type_bits, typedef_bits)
        # The assembler LSB-aligns an operand within its column, so the bits it
        # does not use are the column's high bits.
        if bits < column_bits:
            layout.append(_padding_field(_padding_field_name(field_idx), column_bits - bits))
        layout.append(
            {
                "name": f"{op_name}_{field_idx}",
                "sv_type": sv_type,
                "bits": bits,
                "operand": op_type,
            }
        )
    return layout


def _check_member_names(where: str, layout_fields: list[dict[str, Any]]) -> None:
    names = [f["name"] for f in layout_fields]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate field names {names}")
    keywords = sorted(set(names) & _SV_KEYWORDS)
    if keywords:
        raise ValueError(f"{where}: field names are SystemVerilog keywords: {keywords}")


def _slot_union_descriptors(typedef_bits: dict[str, int]) -> list[dict[str, Any]]:
    """Per-slot union layout structs and per-instruction union members.

    *typedef_bits* maps each generated enum typedef name to its bit width.
    """
    type_bits = get_operand_type_bits()
    slots: list[dict[str, Any]] = []

    for slot_name, slot_union in SLOT_UNIONS.items():
        opcode_key, struct_base = _SLOT_META[slot_name]
        opcode_enum = f"{opcode_key}_t"
        fields: list[dict[str, Any]] = []
        for uf in slot_union.fields:
            fields.append(
                {
                    "index": uf.index,
                    "name": _canonical_field_name(uf.canonical_type, uf.index),
                    "bits": uf.bits,
                    "canonical_type": uf.canonical_type,
                    "sv_type": _sv_logic_type(uf.canonical_type, uf.bits),
                }
            )

        operand_width = sum(f["bits"] for f in fields)
        slot_width = slot_union.opcode_bits + operand_width

        instructions: list[dict[str, Any]] = []
        for inst_name, inst_def in INSTRUCTION_SPEC[slot_name].items():
            layout_fields = _instruction_layout_fields(
                slot_union,
                fields,
                inst_name,
                inst_def,
                type_bits,
                typedef_bits,
            )
            struct_bits = sum(f["bits"] for f in layout_fields)
            if struct_bits != operand_width:
                raise ValueError(
                    f"{slot_name}.{inst_name}: operand layout is {struct_bits} bits, "
                    f"expected operand width {operand_width}"
                )
            _check_member_names(f"{slot_name}.{inst_name}", layout_fields)
            instructions.append(
                {
                    "name": inst_name,
                    "sv_struct": _instruction_struct_name(inst_name),
                    "layout_fields": layout_fields,
                    "struct_bits": struct_bits,
                }
            )
        slots.append(
            {
                "slot": slot_name,
                "opcode_enum": opcode_enum,
                "opcode_width": slot_union.opcode_bits,
                "struct_name": f"{struct_base}_t",
                "union_name": f"{struct_base}_u",
                "width": slot_width,
                "operand_width": operand_width,
                "fields": fields,
                "instructions": instructions,
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


def _check_sv_typedef_map() -> None:
    """Every enum-backed operand type maps to its own generated typedef."""
    stale = sorted(set(_OPERAND_TYPE_TO_SV_TYPEDEF) - VALID_OPERAND_TYPES)
    if stale:
        raise ValueError(f"_OPERAND_TYPE_TO_SV_TYPEDEF has unknown operand types: {stale}")
    for op_type in sorted(VALID_OPERAND_TYPES):
        token_cls = OPERAND_TYPE_MAP[op_type]
        if not issubclass(token_cls, ipu_token.EnumToken):
            continue
        expected = f"{utils.camel_case_to_snake_case(token_cls.__name__)}_t"
        actual = _OPERAND_TYPE_TO_SV_TYPEDEF.get(op_type)
        if actual != expected:
            raise ValueError(
                f"_OPERAND_TYPE_TO_SV_TYPEDEF[{op_type!r}] must be {expected!r}, "
                f"got {actual!r}"
            )


def build_codegen_context(source_commit: str = UNKNOWN_SOURCE_COMMIT) -> dict[str, Any]:
    """Build the Jinja render context from live assembler metadata."""
    _check_sv_typedef_map()
    enum_list = _enum_descriptors_for_templates()
    typedef_bits = {e["sv_type"]: e["width"] for e in enum_list}
    return {
        "enums": {e["name"]: [(m["value"], m["name"]) for m in e["members"]] for e in enum_list},
        "enum_types": enum_list,
        "slots": _slot_union_descriptors(typedef_bits),
        "compound_members": _compound_members(),
        "compound_width": compound_inst.CompoundInst.bits(),
        "source_commit": source_commit,
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


def write_generated_file(
    template_name: str,
    output_path: str | Path,
    context: dict[str, Any] | None = None,
) -> None:
    """Render *template_name* and write to *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_template(template_name, context), encoding="utf-8")


def source_commit_from_status(status_text: str) -> str:
    """Commit recorded by tools/workspace_status.sh, e.g. in Bazel's stable-status.txt."""
    for line in status_text.splitlines():
        key, _, value = line.partition(" ")
        if key == _SOURCE_COMMIT_STATUS_KEY and value:
            return value
    return UNKNOWN_SOURCE_COMMIT


def workspace_source_commit() -> str:
    """Commit of the enclosing checkout, as tools/workspace_status.sh reports it.

    ``bazel run`` changes the working directory, so the source tree it reports
    in ``BUILD_WORKSPACE_DIRECTORY`` takes precedence.
    """
    start = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    try:
        top = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            [str(Path(top) / "tools" / "workspace_status.sh")],
            cwd=top, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN_SOURCE_COMMIT
    return source_commit_from_status(status)


def generate_sv_package(output_path: str | Path, source_commit: str | None = None) -> None:
    """Generate a SystemVerilog package with instruction-format structs and enums.

    *source_commit* defaults to the commit of the enclosing git checkout.
    """
    if source_commit is None:
        source_commit = workspace_source_commit()
    write_generated_file(
        "ipu_instr_pkg.sv.j2", output_path, build_codegen_context(source_commit)
    )
