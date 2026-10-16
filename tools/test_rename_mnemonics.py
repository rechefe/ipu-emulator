"""The mnemonic rename script (E03-19)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rename_mnemonics import rename_text, rename_tree  # noqa: E402

ASM = """\
LDR_MULT_REG r0 lr1 cr2;;
ldr_cyclic_mult_reg lr0 cr1 lr2;;
LDR_MULT_MASK_REG lr0 cr1;;
STR_POST_AAQ_REG cr0 lr1;;
str_acc_reg cr0 lr1;;
INCR_MOD_POW2 lr0 lr1 4;;
blt lr0 lr1 loop;;
BGE lr0 lr1 loop;;
BGT lr0 lr1 loop;;
ble lr0 lr1 loop;;
BKPT;;
"""

RENAMED_ASM = """\
LD.MULT r0 lr1 cr2;;
ld.mult.cyclic lr0 cr1 lr2;;
LD.MASK lr0 cr1;;
ST.AAQ cr0 lr1;;
st.acc cr0 lr1;;
INC.MODP2 lr0 lr1 4;;
blt.s lr0 lr1 loop;;
BGE.S lr0 lr1 loop;;
BGT.S lr0 lr1 loop;;
ble.s lr0 lr1 loop;;
HALT;;
"""


def test_asm_keeps_the_case_style():
    assert rename_text(ASM, ".asm") == (RENAMED_ASM, 11)


def test_renaming_twice_changes_nothing():
    assert rename_text(RENAMED_ASM, ".asm") == (RENAMED_ASM, 0)


def test_prose_renames_upper_case_only():
    text = "Use `BLT` here; a blt in prose is left alone; BLT.S stays.\n"
    assert rename_text(text, ".md") == (
        "Use `BLT.S` here; a blt in prose is left alone; BLT.S stays.\n",
        1,
    )


def test_python_renames_handlers_and_leaves_longer_identifiers():
    text = '"execute_fn": "execute_bkpt", cond_blt = 1, x.BLT, uses_str_acc_reg\n'
    assert rename_text(text, ".py") == (
        '"execute_fn": "execute_halt", cond_blt = 1, x.BLT, uses_str_acc_reg\n',
        1,
    )


def test_a_name_after_a_string_escape_is_renamed():
    text = '"SET lr0 cr8;;\\nBKPT;;\\tblt lr0 lr1 x;;"\n'
    assert rename_text(text, ".py") == ('"SET lr0 cr8;;\\nHALT;;\\tblt.s lr0 lr1 x;;"\n', 2)


def test_a_full_stop_does_not_protect_a_name():
    assert rename_text("It ends with STR_ACC_REG.\n", ".md") == ("It ends with ST.ACC.\n", 1)


def test_a_regex_escaped_dotted_name_is_left_alone():
    text = "r\"mnemonic 'BLT\\.U' is reserved\"\n"
    assert rename_text(text, ".py") == (text, 0)


def test_other_suffixes_are_untouched():
    assert rename_text(ASM, ".json") == (ASM, 0)


def test_tree_skips_planning_and_generated_baselines(tmp_path: Path):
    for path in ("prog.asm", "planning/note.md", "pkg/test/baselines/x.md", "doc.md"):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("BKPT;;\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    assert rename_tree(tmp_path, check=True) == {"doc.md": 1, "prog.asm": 1}
    assert (tmp_path / "prog.asm").read_text() == "BKPT;;\n"
    assert rename_tree(tmp_path) == {"doc.md": 1, "prog.asm": 1}
    assert (tmp_path / "prog.asm").read_text() == "HALT;;\n"
    assert (tmp_path / "planning/note.md").read_text() == "BKPT;;\n"
    assert rename_tree(tmp_path) == {}
