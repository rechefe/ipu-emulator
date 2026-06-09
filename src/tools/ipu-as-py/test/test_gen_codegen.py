"""Tests for instruction-format code generation."""

from pathlib import Path

from ipu_as import gen_codegen
from ipu_as.compound_inst import CompoundInst
from ipu_common.instruction_spec import SLOT_UNIONS


def test_build_context_matches_compound_width():
    ctx = gen_codegen.build_codegen_context()
    assert ctx["compound_width"] == CompoundInst.bits()
    assert len(ctx["inst_bit_fields"]) == len(CompoundInst.get_fields())


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


def test_generate_c_header_matches_historical_shape(tmp_path: Path):
    out = tmp_path / "ipu_inst.h"
    gen_codegen.generate_c_header(out)
    text = out.read_text(encoding="utf-8")
    assert "#ifndef IPU_INST_H" in text
    assert "typedef enum" in text
    assert "ipu_compound_inst_t" in text
    assert "per-slot" not in text.lower()
    assert f"#define IPU_COMPOUND_INST_WIDTH {CompoundInst.bits()}" in text
    for name, _bits in CompoundInst.get_fields():
        assert name in text


def test_generate_sv_package_is_proper_systemverilog(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    assert "package ipu_instr_pkg;" in text
    assert "endpackage : ipu_instr_pkg" in text
    assert "typedef enum logic" in text
    assert "typedef struct packed" in text
    assert "typedef union packed" in text
    assert "xmem_slot_t" in text
    assert "xmem_slot_u" in text
    assert "ipu_compound_inst_t" in text
    assert "ipu_compound_inst_flat_t" in text
    assert f"IPU_COMPOUND_INST_WIDTH = {CompoundInst.bits()}" in text
    # Typedefs use _t suffix; enum literals are sized (e.g. 3'd5)
    assert "_e;" not in text
    assert "lr_reg_field_t" in text
    assert "logic [3:0] lr_idx_2; // dest (MultStageReg)" in text  # LDR_MULT_REG
    assert "3'd" in text or "2'd" in text or "1'd" in text
    assert "break_inst;" in text  # reserved-word-safe union member name
    assert "} break;" not in text
    # One enum member per line
    assert "LR_REG_FIELD_LR0 = " in text
    assert "\n    LR_REG_FIELD_LR1 = " in text


def test_union_members_padded_to_slot_width():
    ctx = gen_codegen.build_codegen_context()
    for slot in ctx["slots"]:
        for inst in slot["instructions"]:
            assert inst["struct_bits"] == slot["width"], (
                f"{slot['slot']}.{inst['name']}: {inst['struct_bits']} != {slot['width']}"
            )


def test_sv_union_matches_slot_field_names(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    # Union members use same wire field names as {slot}_slot_t (union diagram columns)
    assert "cr_idx_0" in text and "lr_idx_1" in text and "lr_idx_2" in text
    # STR_ACC_REG: fields 0,1 used; lr_idx_2 unused (diagram shows — in col 2)
    i = text.index("} str_acc_reg;")
    chunk = text[i - 500 : i]
    assert "cr_idx_0" in chunk and "lr_idx_1" in chunk
    assert "logic [3:0] lr_idx_2;" in chunk
    assert "// offset" in chunk and "// base" in chunk
    # MULT.EE: ra in CrIdx column (field 0), diagram shows ra:MultStageReg
    j = text.index("} mult_ee;")
    ee = text[j - 500 : j]
    assert "logic [3:0] cr_idx_0; // ra (MultStageReg)" in ee
    assert "lr_idx_1; // cyclic_offset" in ee
    assert "lr_idx_2; // mask_shift" in ee
    # MULT.EE.RR: field 2 unused — lr_idx_2 pad in place, not always "field 2 at end"
    k = text.index("} mult_ee_rr;")
    rr = text[k - 400 : k]
    assert "lr_idx_1; // mask_shift" in rr
    assert "logic [3:0] lr_idx_2;" in rr
    assert "mult_mask_offset_immediate_3; // mask_offset" in rr


def test_render_is_deterministic():
    a = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    b = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    assert a == b
