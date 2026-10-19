"""GNU numeric local labels: `1:` defines, `1f` / `1b` refer forward / back (E05-14)."""

from __future__ import annotations

import pytest

from ipu_as import diagnostics
from ipu_as.errors import LabelError
from ipu_as.session import AssemblySession


def _words(source: str) -> tuple[int, ...]:
    return AssemblySession(source).assemble().words


def test_each_macro_expansion_branches_to_its_own_loop() -> None:
    expanded = (
        "{% macro loop() %}\n1:\nNOP;;\nBNE lr0 lr1 1b;;\n{% endmacro %}\n"
        "{{ loop() }}{{ loop() }}HALT;;\n"
    )
    by_name = "a:\nNOP;;\nBNE lr0 lr1 a;;\nb:\nNOP;;\nBNE lr0 lr1 b;;\nHALT;;\n"
    assert _words(expanded) == _words(by_name)


def test_forward_reference_skips_the_current_words_own_label() -> None:
    # Word 0 is `1:` and branches with 1f, so the target is the next `1:` (word 2).
    local = "1:\nB 1f;;\nNOP;;\n1:\nHALT;;\n"
    named = "B x;;\nNOP;;\nx:\nHALT;;\n"
    assert _words(local) == _words(named)


def test_backward_reference_may_target_its_own_word() -> None:
    assert _words("NOP;;\n1:\nB 1b;;\n") == _words("NOP;;\nself:\nB self;;\n")


@pytest.mark.parametrize(
    ("source", "message", "line"),
    [
        ("B 1b;;\n1:\nHALT;;\n", "local label 1b has no definition at or before line 1", 1),
        ("1:\nNOP;;\nB 1f;;\n", "local label 1f has no definition after line 3", 3),
        ("1:\nNOP;;\nB 1;;\n", "use 1f or 1b", 3),
    ],
)
def test_unresolved_local_reference_is_a_located_label_error(source, message, line) -> None:
    with pytest.raises(LabelError, match=message) as info:
        AssemblySession(source).assemble()
    assert info.value.location.line == line
    assert info.value.location.column == 3


def test_named_label_defined_twice_still_fails_with_the_same_message() -> None:
    with pytest.raises(LabelError) as info:
        AssemblySession("a:\nNOP;;\na:\nNOP;;\n").assemble()
    assert str(info.value) == (
        "Label 'a' is defined for the second time at Line 3, Column 1. "
        "Previous definition at Line 1, Column 1."
    )


def test_local_labels_are_not_in_the_named_symbols() -> None:
    source = "1:\nNOP;;\nx:\nNOP;;\n1:\nB 1b;;\n"
    assert dict(AssemblySession(source).assemble().symbols) == {"x": 1}


@pytest.mark.parametrize(
    ("source", "line", "column"),
    [("B 1b;;\n1:\nHALT;;\n", 0, 2), ("1:\nNOP;;\nB 1f;;\n", 2, 2), ("1:\nNOP;;\nB 1;;\n", 2, 2)],
)
def test_check_reports_local_label_errors_at_the_reference(source, line, column) -> None:
    (found,) = diagnostics.check(source)
    assert (found.stage, found.line, found.column) == ("assemble", line, column)
    assert not found.approximate


def test_named_label_shaped_like_a_local_reference_is_rejected() -> None:
    with pytest.raises(LabelError, match="looks like a local label reference") as info:
        AssemblySession("NOP;;\n10b:\nB 10b;;\n").assemble()
    assert info.value.location.line == 2


def test_leading_zeros_name_the_same_local_label() -> None:
    assert _words("01:\nNOP;;\nB 1b;;\n") == _words("1:\nNOP;;\nB 1b;;\n")
