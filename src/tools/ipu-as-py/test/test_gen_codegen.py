"""Tests for instruction-format code generation."""

from pathlib import Path

import pytest

from ipu_as import gen_codegen
from ipu_as.compound_inst import CompoundInst


def test_build_context_matches_compound_width():
    ctx = gen_codegen.build_codegen_context()
    assert ctx["compound"]["width"] == CompoundInst.bits()
    assert len(ctx["compound"]["fields"]) == len(CompoundInst.get_fields())


def test_slot_struct_field_bit_sum():
    ctx = gen_codegen.build_codegen_context()
    for slot in ctx["slots"]:
        field_bits = sum(f["bits"] for f in slot["fields"])
        assert field_bits == slot["width"], slot["slot"]


def test_generate_c_header_contains_guard_and_struct(tmp_path: Path):
    out = tmp_path / "ipu_inst.h"
    gen_codegen.generate_c_header(out)
    text = out.read_text(encoding="utf-8")
    assert "#ifndef IPU_INST_H" in text
    assert "typedef struct" in text
    assert "ipu_compound_inst_t" in text
    assert f"#define IPU_COMPOUND_INST_WIDTH {CompoundInst.bits()}" in text


def test_generate_sv_package_contains_package_and_width(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    gen_codegen.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    assert "package ipu_instr_pkg;" in text
    assert "endpackage : ipu_instr_pkg" in text
    assert "typedef struct packed" in text
    assert f"IPU_COMPOUND_INST_WIDTH = {CompoundInst.bits()}" in text
    assert "MULT_INST_OPCODE" in text or "mult_inst_opcode" in text


def test_render_is_deterministic():
    a = gen_codegen.render_template("ipu_inst.h.j2")
    b = gen_codegen.render_template("ipu_inst.h.j2")
    assert a == b
