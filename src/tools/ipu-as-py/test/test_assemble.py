import pytest

from ipu_as.lark_tree import assemble


def test_smoke():
    code = """
start:
    b +1;
    ;;
"""
    assert len(assemble(code)) == 1


# ---------------------------------------------------------------------------
# Pseudo-instructions: each must assemble to the exact same binary as its
# hand-written real-instruction expansion (zero runtime cost, never appears
# in the binary as a distinct opcode).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pseudo_code, real_code",
    [
        ("BGT lr0 lr1 +1;;", "BLT lr1 lr0 +1;;"),
        ("BGT cr1 lr3 +2;;", "BLT lr3 cr1 +2;;"),
        ("BLE lr0 lr1 +1;;", "BGE lr1 lr0 +1;;"),
        ("BLE cr2 lr4 +2;;", "BGE lr4 cr2 +2;;"),
        ("BZ lr0 +1;;", "BEQ lr0 cr0 +1;;"),
        ("BZ cr3 +2;;", "BEQ cr3 cr0 +2;;"),
        ("BNZ lr0 +1;;", "BNE lr0 cr0 +1;;"),
        ("BNZ cr3 +2;;", "BNE cr3 cr0 +2;;"),
        ("B +1;;", "BEQ cr0 cr0 +1;;"),
        ("B +2;;", "BEQ cr0 cr0 +2;;"),
    ],
)
def test_pseudo_instruction_matches_hand_written_expansion(pseudo_code, real_code):
    assert assemble(pseudo_code) == assemble(real_code)


def test_pseudo_instructions_are_case_insensitive():
    assert assemble("bgt lr0 lr1 +1;;") == assemble("BLT lr1 lr0 +1;;")
    assert assemble("bz lr0 +1;;") == assemble("BEQ lr0 cr0 +1;;")
    assert assemble("b +1;;") == assemble("BEQ cr0 cr0 +1;;")


@pytest.mark.parametrize(
    "code",
    [
        "BGT lr0 lr1;;",  # missing label, BGT needs 3 operands
        "BLE lr0 lr1;;",
        "BZ lr0;;",  # missing label, pseudo BZ needs 2 operands
        "BNZ lr0;;",
        "BZ lr0 lr1 +1;;",  # no real 3-operand BZ anymore; only the 2-operand pseudo exists
        "BNZ lr0 lr1 +1;;",
    ],
)
def test_pseudo_instruction_wrong_arity_raises_clear_error(code, capsys):
    with pytest.raises(SystemExit):
        assemble(code)
    assert "expects" in capsys.readouterr().out
