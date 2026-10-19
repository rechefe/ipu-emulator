"""Generate the SystemVerilog package from the instruction format (E08-13).

Emits the opcode and operand enums, per-slot structs (opcode + operand union)
and per-instruction ``union packed`` views derived from the slot union layouts.
The package describes the hardware word only (E03-22): emulator-only slots,
their instructions and any enum only they use are left out. Every
name comes from ISA data (``SlotInfo.opcode_enum``, ``OperandType.sv_typedef``,
``sv_struct_base``), never from a Python class name, so a refactor of the
assembler cannot rename a downstream RTL type. Reads ``ipu_isa`` only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jinja2

from ipu_isa.operand_types import OPERAND_TYPES, OperandType, snake_name
from ipu_isa.slots import Slot
from ipu_isa.tags import Tag
from ipu_isa.union_layout import SlotUnion
from ipu_isa.view import Instruction, Isa, isa

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: The SV package's first line. Its second names the SHA-256 of everything
#: after the two: the file changes only when its content does, and a vendored
#: copy can be checked (verify_sv_package).
SV_PACKAGE_HEADER = "// Auto-generated from instruction_spec.py — do not edit.\n"
SV_PACKAGE_TEMPLATE = "ipu_instr_pkg.sv.j2"

# IEEE 1800-2017 Annex B reserved keywords. Generated identifiers must avoid them.
_SV_KEYWORDS = frozenset(
    """
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
""".split()
)


def _sv_sized_literal(width: int, value: int) -> str:
    """SystemVerilog sized integer literal, e.g. width=3 value=5 → ``3'd5``."""
    return f"{width}'d{value}"


# Slot name → opcode EnumToken descriptor key and struct basename.
_SV_RESERVED_STRUCT_NAMES = frozenset(
    {
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
    }
)


#: The operand-type enums, in the order the package declares them (after the
#: slot opcode enums). It is the order the generator emitted when it read the
#: assembler's token classes; kept so the package stays byte-identical, so a
#: position here is wire format. Checked at import: it names every operand type
#: with an ``sv_typedef`` exactly once.
OPERAND_ENUM_ORDER = (
    "MultStageReg",
    "LrIdx",
    "CrIdx",
    "DstructureCrIdx",
    "LcrIdx",
    "LrdIdx",
    "ActivationFn",
    "ElementsInRow",
    "HorizontalStride",
    "VerticalStride",
)


def _sanitize_enum_member(name: str) -> str:
    return name.upper().replace(".", "_").replace("-", "_")


def _sv_logic_type(canonical_type: str, bits: int) -> str:
    typedef_name = OPERAND_TYPES[canonical_type].sv_typedef
    if typedef_name is not None:
        return typedef_name
    return f"logic [{bits - 1}:0]"


def _canonical_field_name(canonical_type: str, field_index: int) -> str:
    base = snake_name(canonical_type)
    return f"{base}_{field_index}"


def _instruction_struct_name(inst_name: str) -> str:
    base = _sanitize_enum_member(inst_name).lower()
    if base in _SV_RESERVED_STRUCT_NAMES or base in _SV_KEYWORDS:
        return f"{base}_inst"
    return base


def _operand_sv_type(
    op_type: OperandType,
    column_bits: int,
    typedef_bits: dict[str, int],
) -> tuple[str, int]:
    """SV type and width of an operand declared at its own width.

    Raises if the operand does not fit its union column, or if its SV type is
    wider than the bits the assembler reserves for it.
    """
    # Derived-width immediates (no authored width) are defined as filling the
    # union column they were packed into.
    reserved_bits = op_type.bits or column_bits
    if reserved_bits > column_bits:
        raise ValueError(
            f"{op_type.name} reserves {reserved_bits} bits but its union column is "
            f"{column_bits} bits"
        )
    typedef_name = op_type.sv_typedef
    if typedef_name is None:
        return f"logic [{reserved_bits - 1}:0]", reserved_bits
    bits = typedef_bits[typedef_name]
    if bits > reserved_bits:
        raise ValueError(
            f"{typedef_name} is {bits} bits but the assembler reserves only "
            f"{reserved_bits} bits for {op_type.name}"
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
    slot_union: SlotUnion,
    slot_fields: list[dict[str, Any]],
    instruction: Instruction,
    typedef_bits: dict[str, int],
) -> list[dict[str, Any]]:
    """Operand-area struct members for a per-instruction union member (MSB → LSB).

    The opcode lives outside ``{slot}_slot_u`` in ``{slot}_slot_t`` — it is shared
    across all instructions in the slot.  Each operand is declared at its own
    width and named ``<operand>_<column>``; unused bits of a column become
    ``padding_<column>``, and operand-less instructions use a single ``padding``
    field for the whole payload.
    """
    bindings = dict(slot_union.opcode_bindings.get(instruction.mnemonic, ()))

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
        op_type = instruction.operand(op_name).type
        sv_type, bits = _operand_sv_type(op_type, column_bits, typedef_bits)
        # The assembler LSB-aligns an operand within its column, so the bits it
        # does not use are the column's high bits.
        if bits < column_bits:
            layout.append(_padding_field(_padding_field_name(field_idx), column_bits - bits))
        layout.append(
            {
                "name": f"{op_name}_{field_idx}",
                "sv_type": sv_type,
                "bits": bits,
                "operand": op_type.name,
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


def _slot_union_descriptors(view: Isa, typedef_bits: dict[str, int]) -> list[dict[str, Any]]:
    """Per-slot union layout structs and per-instruction union members.

    *typedef_bits* maps each generated enum typedef name to its bit width.
    """
    slots: list[dict[str, Any]] = []

    for slot in _hardware_slots(view, view.spec_slot_order):
        slot_name, slot_union, info = slot.value, view.layout.unions[slot], view.slots[slot]
        opcode_key, struct_base = info.opcode_enum, info.sv_struct_base
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
        for instruction in view.instructions(slot):
            inst_name = instruction.mnemonic
            layout_fields = _instruction_layout_fields(
                slot_union, fields, instruction, typedef_bits
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


def _compound_members(view: Isa) -> list[dict[str, Any]]:
    """Nested compound struct members in MSB → LSB order (matches encode layout)."""
    members: list[dict[str, Any]] = []
    type_counts: dict[Slot, int] = {}

    order = view.layout.slots_with_tag(Tag.HARDWARE)
    for slot in (s for s in order for _ in range(view.slots[s].count)):
        struct_base = view.slots[slot].sv_struct_base
        sv_type = f"{struct_base}_t"

        count = type_counts.get(slot, 0)
        type_counts[slot] = count + 1
        if view.slots[slot].count > 1:
            member_name = f"{struct_base}_{count}"
        else:
            member_name = struct_base

        members.append(
            {
                "name": member_name,
                "sv_type": sv_type,
                "slot": slot.value,
            }
        )

    return members


def _check_operand_enum_order() -> None:
    typed = {name for name, t in OPERAND_TYPES.items() if t.sv_typedef is not None}
    if set(OPERAND_ENUM_ORDER) != typed or len(OPERAND_ENUM_ORDER) != len(typed):
        raise ValueError(
            "OPERAND_ENUM_ORDER in ipu_gen/sv_package.py must list each operand type "
            f"with an sv_typedef once: missing {sorted(typed - set(OPERAND_ENUM_ORDER))}, "
            f"extra {sorted(set(OPERAND_ENUM_ORDER) - typed)}; a new type's position "
            "is wire format"
        )


_check_operand_enum_order()


def _hardware_slots(view: Isa, order: Iterable[Slot]) -> tuple[Slot, ...]:
    """The hardware-tagged slots of ``order``, in that order.

    Takes the order because the enums and slot structs follow spec order while
    the compound struct follows layout order (``layout.slots_with_tag``).
    """
    return tuple(slot for slot in order if view.slots[slot].tag is Tag.HARDWARE)


def _hardware_operand_types(view: Isa) -> frozenset[str]:
    """Operand types a hardware slot uses, as an operand or as a field's type."""
    used: set[str] = set()
    for slot in _hardware_slots(view, view.layout.layout_order):
        used.update(field.canonical_type for field in view.layout.unions[slot].fields)
        for instruction in view.instructions(slot):
            used.update(op.type.name for op in instruction.operands)
    return frozenset(used)


def _typedef_base(view: Isa, name: str) -> str:
    typedef = view.operand_types[name].sv_typedef
    if typedef is None or not typedef.endswith("_t"):
        raise ValueError(f"operand type {name}: sv_typedef {typedef!r} must end in _t")
    return typedef.removesuffix("_t")


def enum_descriptors(view: Isa | None = None) -> dict[str, list[tuple[int, str]]]:
    """Every SV enum in package order: each hardware slot's opcodes, then the
    operand enums a hardware slot uses.

    Returns:
        ``{name: [(value, MEMBER), ...]}``, where ``name`` is the typedef without
        ``_t``. Opcodes are in spec order (position = opcode); operand members
        in their domain's encoding order.
    """
    view = isa() if view is None else view
    enums: dict[str, list[tuple[int, str]]] = {}
    for slot in _hardware_slots(view, view.spec_slot_order):
        enums[view.slots[slot].opcode_enum] = _members(
            instruction.mnemonic for instruction in view.instructions(slot)
        )
    used = _hardware_operand_types(view)
    for name in (n for n in OPERAND_ENUM_ORDER if n in used):
        enums[_typedef_base(view, name)] = _members(view.operand_types[name].domain.members)
    return enums


def _members(names: Iterable[str]) -> list[tuple[int, str]]:
    return [(i, _sanitize_enum_member(name)) for i, name in enumerate(names)]


def _enum_descriptors_for_templates(view: Isa) -> list[dict[str, Any]]:
    """The enums with precomputed bit-width for SV typedefs."""
    result: list[dict[str, Any]] = []
    for enum_name, members in enum_descriptors(view).items():
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


def _compound_width(view: Isa) -> int:
    """Bits of the hardware word: each hardware slot's opcode and operand union, per copy."""
    return view.layout.word_bits(Tag.HARDWARE)


def _slot_list(view: Isa, slots: Iterable[Slot]) -> str:
    """Slots as the compound comment names them, in the given order: ``lr×3``."""
    return ", ".join(
        slot.value if view.slots[slot].count == 1 else f"{slot.value}×{view.slots[slot].count}"
        for slot in slots
    )


def _compound_slot_order(view: Isa) -> str:
    """The hardware word's slots, MSB first, as its comment names them: ``lr×3``."""
    return _slot_list(view, view.layout.slots_with_tag(Tag.HARDWARE))


def _emulator_slots(view: Isa) -> str:
    """The emulator-only slots, MSB first, that the emulator word appends below bit 0."""
    return _slot_list(view, view.layout.slots_with_tag(Tag.EMULATOR))


def build_codegen_context(view: Isa | None = None) -> dict[str, Any]:
    """Build the Jinja render context from the ISA view (default: ``isa()``)."""
    view = isa() if view is None else view
    enum_list = _enum_descriptors_for_templates(view)
    typedef_bits = {e["sv_type"]: e["width"] for e in enum_list}
    return {
        "enums": {e["name"]: [(m["value"], m["name"]) for m in e["members"]] for e in enum_list},
        "enum_types": enum_list,
        "slots": _slot_union_descriptors(view, typedef_bits),
        "compound_members": _compound_members(view),
        "compound_width": _compound_width(view),
        "compound_slot_order": _compound_slot_order(view),
        "emulator_slots": _emulator_slots(view),
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


def _content_line(body: str) -> str:
    return f"// Content: sha256:{hashlib.sha256(body.encode('utf-8')).hexdigest()}\n"


def render_sv_package(context: dict[str, Any] | None = None, view: Isa | None = None) -> str:
    """The SystemVerilog package: its header line, the hash of its body, the body.

    Rendered from *context* if given, else from *view* (default: ``isa()``).
    """
    if context is not None and view is not None:
        raise TypeError("render_sv_package takes a context or a view, not both")
    body = render_template(
        SV_PACKAGE_TEMPLATE, context if context is not None else build_codegen_context(view)
    )
    return SV_PACKAGE_HEADER + _content_line(body) + body


def verify_sv_package(text: str) -> bool:
    """Whether *text* is an SV package as generated: its body still has the hash
    its header names, so not a byte of it was edited."""
    header, _, rest = text.partition("\n")
    content, _, body = rest.partition("\n")
    return header + "\n" == SV_PACKAGE_HEADER and content + "\n" == _content_line(body)


def generate_sv_package(output_path: str | Path, view: Isa | None = None) -> None:
    """Write the SystemVerilog package (instruction-format structs and enums)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_sv_package(view=view), encoding="utf-8")
