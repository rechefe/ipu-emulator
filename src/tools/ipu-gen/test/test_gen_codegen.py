"""Tests for instruction-format code generation."""

from pathlib import Path

from click.testing import CliRunner

from ipu_as.compound_inst import CompoundInst
from ipu_as.ipu_token import EnumToken
from ipu_gen import layout_svg, sv_package
from ipu_gen.cli import cli
from ipu_isa.layout import slots_with_tag, word_bits
from ipu_isa.slots import SLOTS
from ipu_isa.tags import Tag
from ipu_isa.view import isa

UNIONS = isa().layout.unions


def test_build_context_width_is_the_hardware_word():
    ctx = sv_package.build_codegen_context()
    assert ctx["compound_width"] == word_bits(Tag.HARDWARE) == 155
    # The emulator word appends the emulator-only slots below it (Q41).
    assert CompoundInst.bits() == 186


def test_slot_union_struct_bit_widths():
    ctx = sv_package.build_codegen_context()
    for slot in ctx["slots"]:
        field_bits = slot["opcode_width"] + sum(f["bits"] for f in slot["fields"])
        assert field_bits == slot["width"], slot["slot"]
        su = UNIONS[slot["slot"]]
        assert len(slot["fields"]) == len(su.fields)


def test_compound_members_follow_slot_order():
    ctx = sv_package.build_codegen_context()
    names = [m["name"] for m in ctx["compound_members"]]
    assert names[0].startswith("cond_slot")
    assert names[-1].startswith("store_slot")
    assert sum(1 for n in names if n.startswith("lr_slot")) == 3


def test_generate_sv_package_is_proper_systemverilog(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    sv_package.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    assert "package ipu_instr_pkg;" in text
    assert "endpackage : ipu_instr_pkg" in text
    assert "typedef enum logic" in text
    assert "typedef struct packed" in text
    assert "typedef union packed" in text
    assert "load_slot_t" in text
    assert "load_slot_u" in text
    assert "store_slot_t" in text
    assert "ipu_compound_inst_t" in text
    assert "IPU_COMPOUND_INST_WIDTH = 155;" in text
    assert "opcode_t opcode;" in text
    assert "slot_u operands;" in text
    # Typedefs use _t suffix; enum literals are sized (e.g. 3'd5)
    assert "_e;" not in text
    assert "lr_reg_field_t" in text
    assert "mult_stage_reg_field_t dest_2; // MultStageReg" in text  # LD.MULT
    assert "3'd" in text or "2'd" in text or "1'd" in text
    assert "} break;" not in text
    # One enum member per line
    assert "LR_REG_FIELD_LR0 = " in text
    assert "\n    LR_REG_FIELD_LR1 = " in text


def test_union_members_padded_to_operand_width():
    ctx = sv_package.build_codegen_context()
    for slot in ctx["slots"]:
        operand_width = slot["operand_width"]
        for inst in slot["instructions"]:
            assert inst["struct_bits"] == operand_width, (
                f"{slot['slot']}.{inst['name']}: {inst['struct_bits']} != {operand_width}"
            )


def test_union_member_fields_named_after_their_operands():
    ctx = sv_package.build_codegen_context()
    for slot in ctx["slots"]:
        bindings = UNIONS[slot["slot"]].opcode_bindings
        for inst in slot["instructions"]:
            expected = {f"{op}_{idx}" for idx, op in bindings.get(inst["name"], [])}
            names = {f["name"] for f in inst["layout_fields"] if f["operand"] != "padding"}
            assert names == expected, f"{slot['slot']}.{inst['name']}"


def test_unused_union_columns_named_padding(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    sv_package.generate_sv_package(out)
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
    # LD.MULT: every column used; the 1-bit dest pads the rest of column 2
    j = text.index("} ld_mult;")
    ldr = text[j - 400 : j]
    assert "logic [2:0] padding_2; // padding\n      mult_stage_reg_field_t dest_2;" in ldr
    assert "padding_0" not in ldr and "padding_1" not in ldr


def test_union_members_exclude_opcode(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    sv_package.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    # Opcode is outside the union; union members are operand payload only.
    assert "COND_SLOT_U_WIDTH = 20" in text
    start = text.index("// Operand payload only (opcode is in cond_slot_t.opcode")
    end = text.index("} cond_slot_u;", start)
    union = text[start:end]
    assert "cond_inst_opcode_t opcode" not in union
    nop_start = union.rindex("struct packed {", 0, union.index("} nop;"))
    nop = union[nop_start : union.index("} nop;")]
    assert "opcode" not in nop
    assert "logic [19:0] padding" in nop
    assert "label_0" not in nop


def test_nop_union_members_use_single_padding_field(tmp_path: Path):
    out = tmp_path / "ipu_instr_pkg.sv"
    sv_package.generate_sv_package(out)
    text = out.read_text(encoding="utf-8")
    import re

    # Every slot's NOP member: one padding field, width matches {slot}_slot_u.
    nop_blocks = re.findall(
        r"struct packed \{\n      logic \[(\d+):0\] padding; // padding\n    \} nop;",
        text,
    )
    assert len(nop_blocks) == 7
    widths_by_slot = {
        "load_slot_u": 12,
        "store_slot_u": 8,
        "lr_slot_u": 16,
        "mult_slot_u": 20,
        "acc_slot_u": 13,
        "aaq_slot_u": 8,
        "cond_slot_u": 20,
    }
    for slot_u, width in widths_by_slot.items():
        localparam = f"localparam int unsigned {slot_u.upper()}_WIDTH = {width};"
        assert localparam in text
    assert sorted(int(w) for w in nop_blocks) == sorted(w - 1 for w in widths_by_slot.values())


def test_render_is_deterministic():
    assert sv_package.render_sv_package() == sv_package.render_sv_package()


def test_header_names_the_hash_of_the_body_not_a_commit():
    import hashlib

    first, second, body = sv_package.render_sv_package().split("\n", 2)
    assert first == "// Auto-generated from instruction_spec.py — do not edit."
    assert second == f"// Content: sha256:{hashlib.sha256(body.encode()).hexdigest()}"
    assert "Source:" not in body and "@" not in second


def test_verify_accepts_the_package_and_rejects_any_edit():
    text = sv_package.render_sv_package()
    assert sv_package.verify_sv_package(text)
    at = text.index("package ipu_instr_pkg;")
    edited = text[:at] + "P" + text[at + 1 :]
    assert not sv_package.verify_sv_package(edited)
    assert not sv_package.verify_sv_package(
        text.replace("// Content: sha256:", "// Content: sha256:0", 1)
    )
    assert not sv_package.verify_sv_package("")


def test_compound_comment_lists_the_layout_order():
    text = sv_package.render_sv_package()
    assert "slots MSB → LSB: cond, lr×3, load, mult, acc, aaq, store)" in text
    assert "emulator-only\n  // slots (acc_store, break) below bit 0" in text


def test_no_emulator_only_slot_or_instruction_in_the_package():
    # By hand: the names the ACC_STORE and BREAK slots had in the 186-bit package.
    text = sv_package.render_sv_package()
    for name in (
        "acc_store_slot_t",
        "acc_store_slot_u",
        "break_slot_t",
        "break_slot_u",
        "acc_store_inst_opcode_t",
        "break_inst_opcode_t",
        "ACC_STORE_INST_OPCODE_",
        "BREAK_INST_OPCODE_",
        "} st_acc;",
        "} break_inst;",
        "} break_ifeq;",
        "acc_store_slot;",
        "break_slot;",
    ):
        assert name not in text, name


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def test_renaming_token_classes_does_not_change_the_package():
    """SV names come from ISA data, not from the assembler's class names (E08-13).

    The structural guarantee is the import-graph row ``ipu_gen = ["ipu_isa"]``
    (tools/import_graph.toml); this checks the output as well.
    """
    before = sv_package.render_sv_package()
    # Restored by hand: monkeypatch cannot delete __name__ from a type.
    names = {cls: cls.__name__ for cls in _subclasses(EnumToken)}
    assert len(names) > 20
    try:
        for i, cls in enumerate(names):
            cls.__name__ = f"Renamed{i}"
        assert sv_package.render_sv_package() == before
    finally:
        for cls, name in names.items():
            cls.__name__ = name


def test_enums_match_the_assembler():
    """The SV enums are the assembler's, by name, member and order (an
    independent source: the assembler derives them from its class names)."""
    emulator_opcodes = {SLOTS[slot].opcode_enum for slot in slots_with_tag(Tag.EMULATOR)}
    assembler = {
        name: members
        for name, members in EnumToken.get_all_enum_descriptors().items()
        if name not in emulator_opcodes
    }
    assert sv_package.enum_descriptors() == assembler


def test_cli_writes_the_package(tmp_path: Path):
    out = tmp_path / "pkg.sv"
    result = CliRunner().invoke(cli, ["sv-package", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == sv_package.render_sv_package()


def test_cli_writes_the_docs_pages_and_the_union_svg(tmp_path: Path):
    pages = tmp_path / "pages"
    assert CliRunner().invoke(cli, ["docs-pages", str(pages)]).exit_code == 0
    assert sorted(p.name for p in pages.iterdir()) == [
        "assembly-syntax.md",
        "configuration-registers.md",
        "instructions.md",
        "operand-types.md",
        "programmer-guide.md",
        "registers.md",
    ]
    svg = tmp_path / "union.svg"
    assert CliRunner().invoke(cli, ["union-layout-svg", str(svg)]).exit_code == 0
    assert svg.read_text() == layout_svg.union_layout_svg()
    page = tmp_path / "instruction-format.md"
    assert CliRunner().invoke(cli, ["instruction-format-md", str(page)]).exit_code == 0
    assert "## Hardware word (155 bits)" in page.read_text()


def test_format_page_shows_the_hardware_word_first_then_the_emulator_word(tmp_path: Path):
    from ipu_gen import docs_pages

    page = tmp_path / "instruction-format.md"
    docs_pages.generate_instruction_format_md(page)
    text = page.read_text()
    # By hand: the hardware word is 155 bits, the emulator word 186 (Q41).
    hardware = text.index("## Hardware word (155 bits)")
    emulator = text.index("## Emulator word (186 bits)")
    assert hardware < emulator
    assert word_bits(Tag.HARDWARE) == 155 and word_bits() == 186
    assert "Hardware word Layout - 155 total bits" in text[hardware:emulator]
    assert "Emulator word Layout - 186 total bits" in text[emulator:]
    # By hand, from the SV golden: STORE is the hardware word's lowest slot
    # (9 bits, opcode at bit 8) and COND its highest (opcode at bits 154:152).
    assert ">[8:8]<" in text[hardware:emulator] and ">[154:152]<" in text[hardware:emulator]
    # The emulator-only slots are drawn only in the emulator-word section.
    hardware_part = text[hardware:emulator]
    assert "AccStoreInst" not in hardware_part and "BreakInst" not in hardware_part
    assert "(`ACC_STORE`, `BREAK`) below its bit 0" in text


def test_instruction_reference_tags_every_instruction(tmp_path: Path):
    from ipu_gen import docs_pages

    page = tmp_path / "instructions.md"
    docs_pages.generate_instruction_docs(page)
    text = page.read_text()
    assert "### `HALT` — Halt\n\n**Syntax:** `HALT`\n\n**Tag:** `HARDWARE`" in text
    assert "### `ST.ACC` —" in text
    st_acc = text[text.index("### `ST.ACC` —") :]
    assert "**Tag:** `EMULATOR`" in st_acc.split("###", 2)[1]
    assert "**Tag:** `EMULATOR`" in text[text.index("### `BREAK` —") :].split("###", 2)[1]
    documented = text.count("\n### `")  # NOP's one global entry included
    assert text.count("**Tag:** `") == documented
    guide = tmp_path / "guide.md"
    docs_pages.generate_programmer_guide_md(guide)
    assert guide.read_text().count("**Tag:** `ASSEMBLER`") == 5  # B, BZ, BNZ, BGT.S, BLE.S


def test_cli_verify_sv_package_exit_codes(tmp_path: Path):
    good = tmp_path / "pkg.sv"
    good.write_text(sv_package.render_sv_package(), encoding="utf-8")
    assert CliRunner().invoke(cli, ["verify-sv-package", str(good)]).exit_code == 0
    edited = tmp_path / "edited.sv"
    edited.write_text(good.read_text(encoding="utf-8") + "// edited\n", encoding="utf-8")
    assert CliRunner().invoke(cli, ["verify-sv-package", str(edited)]).exit_code == 1


def test_cli_vscode_data_needs_a_directory_outside_bazel_run(monkeypatch):
    monkeypatch.delenv("BUILD_WORKSPACE_DIRECTORY", raising=False)
    result = CliRunner().invoke(cli, ["vscode-data"])
    assert result.exit_code == 2
    assert "BUILD_WORKSPACE_DIRECTORY" in result.output


def test_cli_resolves_relative_paths_against_the_callers_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))
    result = CliRunner().invoke(cli, ["union-layout-svg", "out/union.svg"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "union.svg").read_text() == layout_svg.union_layout_svg()


def test_timing_table_has_a_row_for_every_rule(tmp_path: Path):
    """The timing model page's table (E09-18): every rule and the slot order label a
    row, and the rows hand-checked against the contract's four rules are there."""
    out = tmp_path / "timing.md"
    assert CliRunner().invoke(cli, ["timing-table", str(out)]).exit_code == 0
    from ipu_gen import docs_pages

    text = out.read_text()
    assert text == docs_pages.timing_contract_table()
    for anchor in ("rule-1", "rule-2", "rule-3", "rule-4", "slot-order"):
        assert f"(#{anchor})" in text
    for row in (
        "| `COND` | `LR` | `reg1`, `reg2`, `reg` | cycle start | [Rule 1](#rule-1) |",
        "| `LOAD` | `LR` | `offset`, `index` | after LR writes | [Rule 1](#rule-1) |",
        "| `MULT` | `R_CYCLIC` | implicit | cycle start | [Rule 2](#rule-2) |",
        "| `MULT` | `R_MASK` | implicit | cycle start | [Rule 2](#rule-2) |",
        "| `ACC` | `MULT_RES` | implicit | same-cycle forward | [Rule 3](#rule-3) |",
        "| `AAQ` | `R_ACC` | implicit | same-cycle forward | [Rule 3](#rule-3) |",
        "| `ACC` | `R_ACC` | implicit | cycle start | [slot order](#slot-order) |",
    ):
        assert row in text, row
    assert "| `LR` | `LR` |" in text and "cycle start | [Rule 4](#rule-4) |" in text


def test_instruction_reference_links_each_read_to_the_timing_model(tmp_path: Path):
    from ipu_gen import docs_pages

    page = tmp_path / "instructions.md"
    docs_pages.generate_instruction_docs(page)
    text = page.read_text()
    entry = text[text.index("### `LD.MULT.CYCLIC` —") :].split("\n### ", 1)[0]
    assert (
        "**Register reads:** `offset` (`LR`): [after LR writes](explanation/timing-model.md#rule-1)"
    ) in entry
    mult = text[text.index("### `MULT.RC.VS` —") :].split("\n### ", 1)[0]
    assert "`R_CYCLIC`: [cycle start](explanation/timing-model.md#rule-2)" in mult
    assert "### `HALT` —" in text
    halt = text[text.index("### `HALT` —") :].split("\n### ", 1)[0]
    assert "Register reads" not in halt
