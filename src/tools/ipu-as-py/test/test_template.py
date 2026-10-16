"""ipu_as.template.render: declared params, defines, a rooted sandboxed loader (E05-09)."""

from __future__ import annotations

import os
from pathlib import Path

import jinja2
import pytest

from ipu_as import diagnostics, template
from ipu_as.errors import TemplateError
from ipu_as.template import Param, ParamSpec, TemplateInputs

HEADER = """\
{#- ipu:params
cpr:        int = 3            # column chunks per row
rows:       int
activation: str = "identity" in (identity, relu, silu)
flags:      bool = false
widths:     tuple[int, ...] = (1, 2)
-#}
{{ cpr }} {{ rows }} {{ activation }} {{ flags }} {{ widths }}"""


def test_header_is_parsed_with_types_defaults_choices_and_lines() -> None:
    spec = template.declared_params(HEADER)
    assert spec.line == 1
    by_name = {param.name: param for param in spec.params}
    assert by_name["cpr"] == Param("cpr", int, 3, None, 2)
    assert by_name["rows"].default is template.REQUIRED
    assert by_name["activation"].choices == ("identity", "relu", "silu")
    assert by_name["flags"].default is False
    assert by_name["widths"] == Param("widths", tuple, (1, 2), None, 6)


def test_defines_are_coerced_from_command_line_strings() -> None:
    rendered = template.render(
        HEADER,
        TemplateInputs(
            defines={"rows": "8", "flags": "true", "widths": "4,5,6", "activation": "relu"}
        ),
    )
    assert rendered == "3 8 relu True (4, 5, 6)"


def test_defaults_render_the_same_as_passing_them() -> None:
    explicit = {"cpr": 3, "rows": 1, "activation": "identity", "flags": False, "widths": (1, 2)}
    assert template.render(HEADER, TemplateInputs(defines={"rows": 1})) == template.render(
        HEADER, TemplateInputs(defines=explicit)
    )


def test_a_param_spec_from_the_caller_replaces_the_header() -> None:
    spec = ParamSpec((Param("n", int, 7),))
    assert template.render("{{ n }}", TemplateInputs(params=spec)) == "7"


@pytest.mark.parametrize(
    ("source", "defines", "message", "line"),
    [
        (HEADER, {"rows": 1, "cprr": 2}, "undeclared define 'cprr'", 1),
        (HEADER, {}, "missing required define 'rows'", 3),
        (HEADER, {"rows": "eight"}, "define 'rows' is 'eight', not of type int", 3),
        (HEADER, {"rows": 1, "activation": "tanh"}, "not one of", 4),
        ("{{ x }}NOP;;", {"x": 1}, "declares no parameters", 1),
        ("{#- ipu:params\nn: float\n-#}", {}, "unknown ipu:params type", 2),
        ("{#- ipu:params\nn int\n-#}", {}, "malformed ipu:params line", 2),
    ],
)
def test_bad_defines_and_headers_are_located_template_errors(source, defines, message, line):
    with pytest.raises(TemplateError, match=message) as info:
        template.render(source, TemplateInputs(defines=defines), source_name="k.asm")
    assert info.value.location.line == line
    assert info.value.location.source == "k.asm"


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "m.asm.j2").write_text(
        "{% macro f() %}NOP;;{% endmacro %}\n{% macro bad() %}\nNOP;;\n{{ 1 + 'a' }}\n{% endmacro %}\n"
    )
    (lib / "body.asm.j2").write_text("NOP;;\nBKPT;;\n")
    (lib / "escape.asm.j2").write_text("{{ ''.__class__.__mro__[1].__subclasses__() }}")
    outside = tmp_path / "secret.asm.j2"
    outside.write_text("NOP;;")
    os.symlink(outside, lib / "link.asm.j2")
    return lib, tmp_path


def test_include_and_import_resolve_in_the_search_path(roots) -> None:
    lib, _ = roots
    assert (
        template.render('{% include "body.asm.j2" %}', TemplateInputs(search_path=(lib,)))
        == "NOP;;\nBKPT;;"
    )
    rendered = template.render(
        '{% import "m.asm.j2" as m %}{{ m.f() }}', TemplateInputs(search_path=(lib,))
    )
    assert rendered == "NOP;;"


@pytest.mark.parametrize(
    "name", ["../secret.asm.j2", "/etc/passwd", "link.asm.j2", "missing.asm.j2"]
)
def test_include_outside_the_search_path_is_refused(roots, name) -> None:
    lib, _ = roots
    with pytest.raises(TemplateError) as info:
        template.render(f'NOP;;\n{{% include "{name}" %}}', TemplateInputs(search_path=(lib,)))
    assert isinstance(info.value.__cause__, jinja2.TemplateNotFound)
    assert info.value.location.line == 2


def test_a_sandbox_escape_inside_an_included_file_is_refused(roots) -> None:
    lib, _ = roots
    with pytest.raises(TemplateError) as info:
        template.render('{% include "escape.asm.j2" %}', TemplateInputs(search_path=(lib,)))
    assert isinstance(info.value.__cause__, jinja2.exceptions.SecurityError)
    assert info.value.location.source == "escape.asm.j2"


def test_an_error_inside_an_included_macro_names_its_file_and_line(roots) -> None:
    lib, _ = roots
    source = 'NOP;;\n{% import "m.asm.j2" as m %}\n{{ m.bad() }}\nBKPT;;\n'
    with pytest.raises(TemplateError) as info:
        template.render(source, TemplateInputs(search_path=(lib,)))
    assert (info.value.location.source, info.value.location.line) == ("m.asm.j2", 4)
    (found,) = diagnostics.check(source, search_path=[lib])
    assert found.stage == "template"
    assert found.line == 2  # the call site in the open file
    assert "(in m.asm.j2:4)" in found.message


def test_an_assemble_error_from_an_included_file_names_it(roots) -> None:
    lib, _ = roots
    (lib / "wrong.asm.j2").write_text("NOP;;\nACC.ADD LR0;;\n")
    source = 'NOP;;\n{% include "wrong.asm.j2" %}\nBKPT;;\n'
    (found,) = diagnostics.check(source, search_path=[lib])
    assert found.stage == "assemble"
    assert found.line == 1  # the include line
    assert "(in wrong.asm.j2:2)" in found.message


def test_a_source_without_markers_is_returned_unrendered() -> None:
    text = "NOP;;\nBKPT;;\n"
    assert template.render(text) is text


def test_a_hash_inside_a_quoted_default_is_part_of_the_value() -> None:
    spec = template.declared_params('{#- ipu:params\nmark: str = "a#b"  # note\n-#}')
    assert spec.params[0].default == "a#b"


def test_a_quoted_choice_may_hold_commas_and_parentheses() -> None:
    spec = template.declared_params('{#- ipu:params\nmode: str = "a,b" in ("a,b", "(c)", d)\n-#}')
    assert spec.params[0].choices == ("a,b", "(c)", "d")


def test_a_parameter_declared_twice_is_an_error() -> None:
    with pytest.raises(TemplateError, match="declares 'n' twice") as info:
        template.declared_params("{#- ipu:params\nn: int = 1\nn: int = 2\n-#}")
    assert info.value.location.line == 3


def test_a_deleted_include_is_reported_as_not_found(tmp_path: Path) -> None:
    (tmp_path / "gone.asm.j2").write_text("NOP;;")
    source = '{% include "gone.asm.j2" %}'
    assert template.render(source, TemplateInputs(search_path=(tmp_path,))) == "NOP;;"
    (tmp_path / "gone.asm.j2").unlink()
    with pytest.raises(TemplateError) as info:
        template.render(source, TemplateInputs(search_path=(tmp_path,)))
    assert isinstance(info.value.__cause__, jinja2.TemplateNotFound)


def test_the_cache_does_not_bypass_define_types() -> None:
    from ipu_as import session

    source = "{#- ipu:params\nx: int\n-#}\n{% for _ in range(x) %}NOP;;\n{% endfor %}BKPT;;\n"
    session.assemble(source, defines={"x": 1})
    with pytest.raises(TemplateError, match="not of type int"):
        session.assemble(source, defines={"x": True})


def test_an_apostrophe_in_a_note_does_not_swallow_it() -> None:
    spec = template.declared_params("{#- ipu:params\nm: str = don't  # a note\n-#}")
    assert spec.params[0].default == "don't"


def test_an_undefined_name_is_a_template_error_naming_it_and_its_line() -> None:
    # StrictUndefined (E05-11): `{{ lable }}` used to render as "" inside the
    # instruction, and the error surfaced as an operand count far from the cause.
    with pytest.raises(TemplateError, match="'lable' is undefined") as info:
        template.render("BKPT;;\nBEQ LR0 LR1 {{ lable }};;\n", source_name="k.asm")
    assert (info.value.location.line, info.value.location.source) == (2, "k.asm")
