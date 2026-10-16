"""AssemblySession: one pipeline, its own symbol table, one render (E05-05)."""

from __future__ import annotations

import warnings

import pytest

import ipu_as.template as template
from ipu_as import session
from ipu_as.errors import LabelError, ParseError, TemplateError
from ipu_as.session import AssemblySession

LOOP = "top:\nNOP;;\nB {label};;\n"


def test_interleaved_sessions_with_the_same_labels_do_not_interfere() -> None:
    first = AssemblySession("a:\nNOP;;\nb:\nB a;;\n")
    second = AssemblySession("NOP;;\nNOP;;\na:\nB a;;\n")
    first.parse()
    second.parse()
    first_words = first.assemble().words
    second_words = second.assemble().words
    assert dict(first.assemble().symbols) == {"a": 0, "b": 1}
    assert dict(second.assemble().symbols) == {"a": 2}
    assert first_words == AssemblySession("a:\nNOP;;\nb:\nB a;;\n").assemble().words
    assert second_words[2] == AssemblySession("NOP;;\nNOP;;\na:\nB +0;;\n").assemble().words[2]


def test_label_defined_in_one_session_is_undefined_in_another() -> None:
    AssemblySession(LOOP.format(label="top")).assemble()
    with pytest.raises(LabelError):
        AssemblySession("B top;;\n").assemble()


def test_a_render_that_yields_braces_is_rendered_once(monkeypatch) -> None:
    calls = []
    real_render = template.render

    def counting_render(text: str) -> str:
        calls.append(text)
        return real_render(text)

    monkeypatch.setattr(template, "render", counting_render)
    source = "{{ '{{' }}\nBKPT;;\n"
    with pytest.raises(ParseError) as info:
        AssemblySession(source).assemble()
    assert len(calls) == 1
    # The parser saw the literal "{{", not a second render's empty string.
    assert "{" in str(info.value)


def test_render_runs_once_per_session(monkeypatch) -> None:
    calls = []
    real_render = template.render
    monkeypatch.setattr(template, "render", lambda text: calls.append(text) or real_render(text))
    assembly = AssemblySession("{% if true %}NOP;;{% endif %}\nBKPT;;\n")
    assembly.render()
    assembly.parse()
    assembly.assemble()
    assert len(calls) == 1


def test_template_failure_is_a_template_error_with_the_jinja_cause() -> None:
    with pytest.raises(TemplateError) as info:
        AssemblySession("{{ 1 + 'a' }}\nBKPT;;\n").assemble()
    assert isinstance(info.value.__cause__, TypeError)


def test_defines_are_not_supported_yet() -> None:
    with pytest.raises(TemplateError, match="defines not supported yet"):
        AssemblySession("BKPT;;\n", defines={"ROWS": 8}).assemble()


def test_result_text_formats() -> None:
    result = AssemblySession("NOP;;\nBKPT;;\n").assemble()
    assert result.rendered == "NOP;;\nBKPT;;\n"
    lines = result.to_mem().splitlines()
    assert len(lines) == 2 and all(line.startswith("0x0") for line in lines)
    assert len(result.to_bin()) == 2 * session.instruction_aligned_bytes_len()


def test_cache_returns_the_same_result_for_the_same_source() -> None:
    source = "cached:\nNOP;;\nB cached;;\n"
    assert session.assemble(source) is session.assemble(source)
    assert session.assemble(source) is not session.assemble(source + "NOP;;\n")


def test_simulation_only_slot_warning_is_recorded_and_raised_on_every_call() -> None:
    source = "STR_ACC_REG lr0 cr0;;\nBKPT;;\n"
    for _ in range(2):  # the second call is a cache hit
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = session.assemble(source)
        assert result.warnings == tuple(str(w.message) for w in caught)
        assert any("simulation-only acc_store slot" in w for w in result.warnings)
