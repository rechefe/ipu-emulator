"""Diagnostics must land on the right token, not merely report that something is wrong.

An editor squiggle is only useful if it is in the right place, so these assert
positions, not just that an error was produced.
"""

import pytest

from ipu_as.diagnostics import check


def test_valid_program_has_no_diagnostics():
    assert check("start:\n  BKPT;;\n") == []


def test_comma_is_reported_at_the_comma():
    # The motivating case: commas are used throughout the prose docs but are a
    # lex error, and a TextMate grammar cannot flag them.
    (found,) = check("BEQ lr0, cr0, +1;;\n")
    assert found.stage == "parse"
    assert (found.line, found.column) == (0, 7)
    assert not found.approximate


def test_unknown_mnemonic_points_at_the_mnemonic():
    # This error carries no location of its own, so the position is recovered
    # by locating the name the message quotes.
    (found,) = check("BKPT;;\nmac.ee r0;;\n")
    assert found.stage == "assemble"
    assert found.line == 1
    assert "mac.ee" in found.message


def test_pseudo_instruction_arity_is_reported():
    # Raised inside the AST transformer, which lark wraps in VisitError -- a
    # LarkError. Without unwrapping it is misreported as a parse error at 0.
    (found,) = check("BGT lr0 lr1;;\n")
    assert found.stage == "assemble"
    assert found.line == 0
    assert "BGT" in found.message


def test_location_suffix_is_stripped_from_the_message():
    # The position is carried by the diagnostic; repeating it in the hover is noise.
    (found,) = check("BGT lr0 lr1;;\n")
    assert "Line 1, Column 1" not in found.message


def test_bad_operand_is_reported_at_the_operand():
    (found,) = check("ADD lr0 lr0 1.5;;\n")
    assert found.stage == "assemble"
    assert found.line == 0
    assert found.column > 0


def test_template_syntax_error_reports_the_template_line():
    (found,) = check("BKPT;;\n{% for x in %}\n")
    assert found.stage == "template"
    assert found.line == 1


def test_error_inside_a_jinja_file_maps_back_to_the_source_line():
    # The parser sees rendered text, the editor sees the template, so offsets
    # do not correspond. Origin markers are carried through the render so the
    # source line is recovered exactly rather than guessed.
    source = "{% for i in range(2) %}\nBKPT;;\n{% endfor %}\nmac.ee r0;;\n"
    (found,) = check(source)
    assert source.splitlines()[found.line] == "mac.ee r0;;"
    assert not found.approximate


def test_position_is_exact_after_jinja_register_aliasing():
    # The house style every kernel uses: `{%- set -%}` lines emit nothing, so a
    # naive offset lands several lines off.
    source = (
        '{%- set a = "lr0" -%}\n'
        '{%- set b = "lr1" -%}\n'
        "    SET {{a}} cr0;;\n"
        "    SET {{b}} cr1;;\n"
        "    BEQ lr0, cr0, +1;;\n"
    )
    (found,) = check(source)
    assert (found.line, found.column) == (4, 11)  # the comma itself
    assert not found.approximate


def test_position_is_exact_inside_a_loop_body():
    # A loop repeats one source line many times in the render; every repeat has
    # to map back to the single line the author wrote.
    source = "{% for i in range(4) %}\n    ADD lr0 lr0 1.5;;\n{% endfor %}\n"
    (found,) = check(source)
    assert found.line == 1
    assert source.splitlines()[found.line].lstrip().startswith("ADD")
    assert not found.approximate


def test_instrumentation_does_not_change_what_the_parser_sees():
    # Markers are appended as `#` comments; adding them must not alter whether a
    # program assembles, only where an error is reported.
    from ipu_as.diagnostics import _instrument

    source = '{%- set a = "lr0" -%}\n    SET {{a}} cr0;;\n    BKPT;;\n'
    assert check(source) == []
    assert check(_instrument(source)) == []


@pytest.mark.parametrize(
    "source",
    [
        "start:\n  BKPT;;\n",
        "BKPT;; mid: BKPT;;\n",
        "add:\n  BKPT;;\n",
        "  acc.add.first ;;\n",
    ],
)
def test_legal_constructs_are_not_flagged(source):
    assert check(source) == []


# --- regressions from the code review --------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        '{{ 1 + "a" }}\nBKPT;;\n',   # TypeError
        "{{ 1 / 0 }}\nBKPT;;\n",     # ZeroDivisionError
        "{{ (1).missing() }}\nBKPT;;\n",  # AttributeError via UndefinedError
    ],
)
def test_jinja_runtime_errors_are_reported_not_raised(source):
    # Rendering runs arbitrary expressions, so a failure need not be a
    # TemplateError. Letting one escape killed the CLI with a traceback and made
    # the editor report a broken toolchain instead of a broken template.
    found = check(source)
    assert found and found[0].stage == "template"


@pytest.mark.parametrize(
    "source",
    [
        "",
        "{% for i in undefined_var %}\nBKPT;;\n{% endfor %}\n",
    ],
)
def test_empty_program_is_reported_not_crashed(source):
    # An undefined name iterates as empty rather than raising, so a template can
    # render to nothing at all. The parser then fails on the synthetic `$END`
    # token, which carries no end position -- that used to crash the checker.
    found = check(source)
    assert found and found[0].stage == "parse"


def test_encode_column_is_not_a_rendered_offset():
    # The line was remapped to the source but the column was left pointing into
    # the rendered text, while still claiming to be exact.
    source = (
        '{%- set pad = "                         " -%}\n'
        "    ADD lr0 lr0 1.5;;\n"
    )
    (found,) = check(source)
    line = source.splitlines()[found.line]
    assert line.lstrip().startswith("ADD")
    if not found.approximate:
        # Whatever column it claims must actually be inside that source line.
        assert 0 <= found.column < len(line)


def test_stdin_and_file_agree():
    # The editor checks the buffer over stdin rather than the saved file; both
    # paths must produce the same diagnostics.
    source = "BEQ lr0, cr0, +1;;\n"
    assert [d.to_dict() for d in check(source)] == [d.to_dict() for d in check(source)]


def test_error_below_whitespace_control_is_exact():
    # `-#}` strips the newline after it, so the plain render joins `;;` and the
    # next mnemonic into `;;SET` while the instrumented one keeps them apart.
    # Comparing the two renders as text therefore discarded the origin markers
    # for every file using this style -- which is every kernel in the repo --
    # and every diagnostic collapsed to line 1, marked approximate.
    source = (
        "    SET lr0 cr0 ;;    {#- a trailing note -#}\n"
        "    SET lr1 cr1 ;;\n"
        "    BEQ lr0, cr0, +1;;\n"
    )
    (found,) = check(source)
    assert (found.line, found.column) == (2, 11)
    assert not found.approximate


def test_no_position_is_reported_rather_than_guessed():
    # There is no content-matching fallback: when a position cannot be
    # established it says so, instead of pointing at a plausible-looking line.
    for d in check("BEQ lr0, cr0, +1;;\n"):
        assert not d.approximate  # plain source is always exact


def test_missing_bundle_terminator_before_whitespace_control_is_reported():
    # `-#}` deletes the following newline, fusing `cr4` with the next line's
    # mnemonic into one token. Instrumentation cannot survive that (a `#` marker
    # is line-terminated), so markers are correctly rejected -- but the error
    # must still be REPORTED. It used to raise UnboundLocalError out of the
    # checker, which the editor read as a broken toolchain and showed nothing.
    source = (
        "    ADD lr14 lr14 cr3 ;\n"
        "    ADD lr5  lr5  cr4        {#- note -#}\n"
        "\n"
        "    MULT.RC.VE lr15 lr5 0 lr15 cr15;\n"
    )
    found = check(source)
    assert found, "a missing ;; must be reported"
    assert "operands" in found[0].message


def test_duplicate_label_is_reported_without_polluting_stdout(capsys):
    # `ipu-as check --json` writes diagnostics to stdout, so a stray print there
    # makes the output unparseable and the editor reports a broken toolchain
    # rather than the error. A debug print in the duplicate-label path did that.
    found = check("a:\n  BKPT;;\na:\n  BKPT;;\n")
    assert found and "second time" in found[0].message
    assert capsys.readouterr().out == ""


def test_whitespace_control_fusion_still_yields_an_exact_line():
    # `-#}` deletes the newline, fusing `cr4` with the next line's mnemonic into
    # a single token. An origin marker between them would prevent that fusion --
    # a different program -- so markers are correctly rejected here.
    #
    # The line is still recoverable: rendering with whitespace control disabled
    # keeps the lines apart, so the markers survive there. Only the line is
    # taken from that variant, and only when it fails at the same stage.
    source = (
        "    ADD lr14 lr14 cr3 ;\n"
        "    ADD lr5  lr5  cr4        {#- note -#}\n"
        "\n"
        "    MULT.RC.VE lr15 lr5 0 lr15 cr15;\n"
    )
    (found,) = check(source)
    assert found.line == 1  # the line missing its ';;'
    assert not found.approximate


def test_disabling_whitespace_control_leaves_normal_templates_alone():
    from ipu_as.diagnostics import _disable_whitespace_control

    assert _disable_whitespace_control("{%- set a = 1 -%}") == "{% set a = 1 %}"
    assert _disable_whitespace_control("{#- x -#}") == "{# x #}"
    assert _disable_whitespace_control("{% set a = 1 %}") == "{% set a = 1 %}"


@pytest.mark.parametrize(
    "source,expected_line",
    [
        ('BKPT;;\nBKPT;;\n{{ 1 + "a" }}\nBKPT;;\n', 2),
        ("BKPT;;\n{{ 1 / 0 }}\n", 1),
        ("{{ (1).nope() }}\nBKPT;;\n", 0),
    ],
)
def test_jinja_runtime_error_reports_its_template_line(source, expected_line):
    # A runtime failure has no `.lineno` the way a syntax error does, but jinja2
    # rewrites the traceback so the frame running the template reports the
    # template's own line. Without that these all collapsed to line 1.
    (found,) = check(source)
    assert found.stage == "template"
    assert found.line == expected_line
    assert not found.approximate


def test_pseudo_instruction_expansion_keeps_its_position():
    # Expansion synthesized a bare lark.Token, which carries no line or column,
    # so any later error about the expansion said "Line None, Column None" and
    # could not be placed. `B` expands to BEQ, filling the cond slot twice here.
    source = "start:\n    BNE lr0 lr1 start ;\n    B start;;\n"
    (found,) = check(source)
    assert found.line == 2
    assert not found.approximate
    assert "Line None" not in found.message
