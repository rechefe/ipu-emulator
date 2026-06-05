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
    assert "logic [3:0] dest" in text  # MultStageReg in 4-bit union field
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


def test_sv_union_padding_in_union_field_positions(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    # STR_ACC_REG uses fields 0,1; field 2 is unused → pad before end of struct
    assert "str_acc_reg" in text
    i = text.index("} str_acc_reg;")
    chunk = text[i - 400 : i]
    assert "offset" in chunk and "base" in chunk
    assert "__pad_2" in chunk
    assert chunk.index("base") < chunk.index("__pad_2")
    # XMEM_NOP: opcode then three in-place union field pads
    j = text.index("} xmem_nop;")
    nop_chunk = text[j - 300 : j]
    assert "__pad_0" in nop_chunk and "__pad_1" in nop_chunk and "__pad_2" in nop_chunk


def test_render_is_deterministic():
    a = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    b = gen_codegen.render_template("ipu_instr_pkg.sv.j2")
    assert a == b
