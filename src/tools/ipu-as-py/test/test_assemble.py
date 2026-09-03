import pytest
from jinja2.exceptions import SecurityError

import ipu_as.template as template
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


# ---------------------------------------------------------------------------
# The Jinja layer is a preprocessor, not an interpreter: a source is rendered
# before it is parsed, so an unsandboxed render makes assembling a file the
# same thing as running whatever the file's author put in it.
# ---------------------------------------------------------------------------


def test_template_escape_is_refused_and_does_not_run(tmp_path):
    marker = tmp_path / "executed"
    code = (
        "{{ self.__init__.__globals__.__builtins__.__import__('os')"
        ".popen('touch " + str(marker) + "').read() }}BKPT;;\n"
    )
    with pytest.raises(SecurityError):
        assemble(code)
    assert not marker.exists()


@pytest.mark.parametrize(
    "code",
    [
        "{% for c in ''.__class__.__mro__ %}\nBKPT;;\n{% endfor %}\n",
        "{{ ({}).__class__.__base__.__subclasses__() }}BKPT;;\n",
    ],
)
def test_attribute_traversal_out_of_the_sandbox_is_refused(code):
    with pytest.raises(SecurityError):
        assemble(code)


def test_unsafe_attribute_yields_nothing_rather_than_the_object():
    # Reading an unsafe attribute is answered with an undefined value, not with
    # the object: printing it yields nothing and using it raises. So this is not
    # an error, it is a program with a hole where the expression was, and
    # nothing of the interpreter reaches the assembled text. (Jinja drops the
    # trailing newline, as it did before the sandbox.)
    assert template.render("{{ ''.__class__ }}\nBKPT;;\n") == "\nBKPT;;"


@pytest.mark.parametrize(
    "code",
    [
        # What the kernels actually use: set, loops over range, filters and
        # whitespace control. The sandbox must not cost any of it.
        '{%- set a = "lr0" -%}\n    SET {{ a }} cr0;;\n',
        "{% for i in range(2) %}\n    BKPT;;\n{% endfor %}\n",
        '    SET {{ "LR0"|lower }} cr0;;\n',
    ],
)
def test_ordinary_templates_still_assemble(code):
    assert assemble(code)
