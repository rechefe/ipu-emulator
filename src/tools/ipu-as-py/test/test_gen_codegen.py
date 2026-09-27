"""Tests for instruction-format code generation."""

from pathlib import Path

from ipu_as import gen_codegen
from ipu_as.compound_inst import CompoundInst
from ipu_common.instruction_spec import SLOT_UNIONS


def test_build_context_matches_compound_width():
    ctx = gen_codegen.build_codegen_context()
    assert ctx["compound_width"] == CompoundInst.bits()


def test_slot_union_struct_bit_widths():
    ctx = gen_codegen.build_codegen_context()
    for slot in ctx["slots"]:
        field_bits = slot["opcode_width"] + sum(f["bits"] for f in slot["fields"])
        assert field_bits == slot["width"], slot["slot"]
        su = SLOT_UNIONS[slot["slot"]]
        assert len(slot["fields"]) == len(su.fields)


def test_compound_members_follow_slot_order():
    ctx = gen_codegen.build_codegen_context()
    names = [m["name"] for m in ctx["compound_members"]]
    assert names[0].startswith("cond_slot")
    assert names[-1].startswith("break_slot")
    assert sum(1 for n in names if n.startswith("lr_slot")) == 3


def test_generate_sv_package_is_proper_systemverilog(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    assert "package ipu_instr_pkg;" in text
    assert "endpackage : ipu_instr_pkg" in text
    assert "typedef enum logic" in text
    assert "typedef struct packed" in text
    assert "typedef union packed" in text
    assert "load_slot_t" in text
    assert "load_slot_u" in text
    assert "store_slot_t" in text
    assert "acc_store_slot_t" in text
    assert "ipu_compound_inst_t" in text
    assert f"IPU_COMPOUND_INST_WIDTH = {CompoundInst.bits()}" in text
    assert "opcode_t opcode;" in text
    assert "slot_u operands;" in text
    # Typedefs use _t suffix; enum literals are sized (e.g. 3'd5)
    assert "_e;" not in text
    assert "lr_reg_field_t" in text
    assert "mult_stage_reg_field_t dest_2; // MultStageReg" in text  # LDR_MULT_REG
    assert "3'd" in text or "2'd" in text or "1'd" in text
    assert "break_inst;" in text  # reserved-word-safe union member name
    assert "} break;" not in text
    # One enum member per line
    assert "LR_REG_FIELD_LR0 = " in text
    assert "\n    LR_REG_FIELD_LR1 = " in text


def test_union_members_padded_to_operand_width():
    ctx = gen_codegen.build_codegen_context()
    for slot in ctx["slots"]:
        operand_width = slot["operand_width"]
        for inst in slot["instructions"]:
            assert inst["struct_bits"] == operand_width, (
                f"{slot['slot']}.{inst['name']}: {inst['struct_bits']} != {operand_width}"
            )


def test_union_member_fields_named_after_their_operands():
    ctx = gen_codegen.build_codegen_context()
    for slot in ctx["slots"]:
        bindings = SLOT_UNIONS[slot["slot"]].opcode_bindings
        for inst in slot["instructions"]:
            expected = {f"{op}_{idx}" for idx, op in bindings.get(inst["name"], [])}
            names = {
                f["name"] for f in inst["layout_fields"] if f["operand"] != "padding"
            }
            assert names == expected, f"{slot['slot']}.{inst['name']}"


def test_unused_union_columns_named_padding(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    # COND BR: reg in column 1; columns 0 and 2 unused → padding_0, padding_2
    marker = "} br;\n    struct packed {\n      logic [19:0] padding"
    pos = text.index(marker)
    start = text.rindex("struct packed {", 0, pos)
    br = text[start:pos]
    assert "padding_0; // padding" in br
    assert "lcr_reg_field_t reg_1; // LcrIdx" in br
    assert "padding_2; // padding" in br
    assert "label_0" not in br
    # LDR_MULT_REG: every column used; the 1-bit dest pads the rest of column 2
    j = text.index("} ldr_mult_reg;")
    ldr = text[j - 400 : j]
    assert "logic [2:0] padding_2; // padding\n      mult_stage_reg_field_t dest_2;" in ldr
    assert "padding_0" not in ldr and "padding_1" not in ldr


def test_union_members_exclude_opcode(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    # Opcode is outside the union; union members are operand payload only.
    assert "COND_SLOT_U_WIDTH = 20" in text
    start = text.index("// Operand payload only (opcode is in cond_slot_t.opcode")
    end = text.index("} cond_slot_u;", start)
    union = text[start:end]
    assert "cond_inst_opcode_t opcode" not in union
    nop_start = union.rindex("struct packed {", 0, union.index("} nop;"))
    nop = union[nop_start:union.index("} nop;")]
    assert "opcode" not in nop
    assert "logic [19:0] padding" in nop
    assert "label_0" not in nop


def test_nop_union_members_use_single_padding_field(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    import re

    # Every slot's NOP member: one padding field, width matches {slot}_slot_u.
    nop_blocks = re.findall(
        r"struct packed \{\n      logic \[(\d+):0\] padding; // padding\n    \} nop;",
        text,
    )
    assert len(nop_blocks) == 9
    widths_by_slot = {
        "load_slot_u": 12,
        "store_slot_u": 8,
        "acc_store_slot_u": 8,
        "lr_slot_u": 16,
        "mult_slot_u": 20,
        "acc_slot_u": 13,
        "aaq_slot_u": 8,
        "cond_slot_u": 20,
        "break_slot_u": 20,
    }
    for slot_u, width in widths_by_slot.items():
        localparam = f"localparam int unsigned {slot_u.upper()}_WIDTH = {width};"
        assert localparam in text
    assert sorted(int(w) for w in nop_blocks) == sorted(w - 1 for w in widths_by_slot.values())


def test_render_is_deterministic():
    a = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    b = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    assert a == b
