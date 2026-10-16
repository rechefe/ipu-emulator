"""The glossary adoption check (tools/glossary_check.py) and its tests.

`test_tree_matches_baseline` is the check itself, run on the hand-written docs
in the runfiles tree. The other tests run the check on small trees in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import glossary_check

_GLOSSARY = """\
# Test glossary

## Program

**Instruction word**:
One VLIW word fetched per cycle.
_Avoid_: VLIW word, compound instruction, bundle, line

**Instruction**:
One operation in one slot.
_Avoid_: inst, op (in prose)

**pc**:
The index of the instruction word.
_Avoid_: program counter, PC (in code)
"""


def _tree(tmp_path: Path, pages: dict[str, str], glossary: str = _GLOSSARY) -> Path:
    (tmp_path / "GLOSSARY.md").write_text(glossary, encoding="utf-8")
    (tmp_path / "tools").mkdir()
    for name, text in pages.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    glossary_check.write_baseline(tmp_path)
    return tmp_path


def _messages(root: Path) -> str:
    return "\n".join(v.message for v in glossary_check.check(root))


def test_tree_matches_baseline() -> None:
    violations = glossary_check.check(Path("."))
    assert not violations, "\n".join(v.message for v in violations)


def test_glossary_has_avoid_lists() -> None:
    terms = glossary_check.parse_glossary(Path("GLOSSARY.md").read_text(encoding="utf-8"))
    assert terms, "no _Avoid_ terms parsed from GLOSSARY.md"


def test_skip_list_names_glossary_terms() -> None:
    """Every skipped term is still listed under some _Avoid_ in GLOSSARY.md."""
    text = Path("GLOSSARY.md").read_text(encoding="utf-8").lower()
    for term in glossary_check.SKIPPED_TERMS:
        assert term in text, f"skipped term {term!r} is not in GLOSSARY.md; drop it"


def test_generated_baseline_passes(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"docs/content/page.md": "One VLIW word per cycle.\n"})
    assert glossary_check.check(root) == []
    baseline = json.loads((root / glossary_check.BASELINE).read_text(encoding="utf-8"))
    assert baseline == {"docs/content/page.md": {"VLIW word": 1}}


def test_added_use_fails_and_names_the_glossary_term(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"docs/content/page.md": "One VLIW word per cycle.\n"})
    (root / "docs/content/page.md").write_text(
        "One VLIW word per cycle.\n\nAnother VLIW word.\n", encoding="utf-8"
    )
    messages = _messages(root)
    assert "'VLIW word'" in messages
    assert "'Instruction word'" in messages
    assert "docs/content/page.md:3" in messages


def test_use_in_new_file_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"README.md": "Clean.\n"})
    (root / "docs/content").mkdir(parents=True)
    (root / "docs/content/new.md").write_text("The program counter advances.\n", encoding="utf-8")
    messages = _messages(root)
    assert "docs/content/new.md:1" in messages
    assert "'pc'" in messages


def test_removed_use_asks_to_lower_the_baseline(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"README.md": "A bundle.\n\nA second bundle.\n"})
    (root / "README.md").write_text("A bundle.\n\nAn instruction word.\n", encoding="utf-8")
    messages = _messages(root)
    assert "lower the baseline" in messages
    assert glossary_check.BASELINE_COMMAND in messages
    glossary_check.write_baseline(root)
    assert glossary_check.check(root) == []


@pytest.mark.parametrize(
    "text",
    [
        "```\nOne VLIW word.\n```\n",
        "~~~~text\nOne VLIW word.\n~~~~\n",
        "- item\n\n  ```asm\n  ; a VLIW word\n  ```\n",
        "Use `VLIW word` here.\n",
        "Use ``a `VLIW word` b`` here.\n",
        "See [the page](bundle.md).\n",
    ],
)
def test_code_and_link_targets_are_not_counted(tmp_path: Path, text: str) -> None:
    terms = glossary_check.parse_glossary(_GLOSSARY)
    assert glossary_check.find_uses(text, terms) == {}


def test_matching_is_case_insensitive_whole_word_and_spans_lines() -> None:
    terms = glossary_check.parse_glossary(_GLOSSARY)
    text = "A vliw\nWord.\nBundles.\nBundled.\nUnbundle.\nA compound instruction.\n"
    assert glossary_check.find_uses(text, terms) == {
        "VLIW word": [1],
        "bundle": [3],
        "compound instruction": [6],
    }


def test_qualifiers_and_skip_list() -> None:
    terms = {t.term for t in glossary_check.parse_glossary(_GLOSSARY)}
    # `line` and `inst` are in the skip list; `op (in prose)` is checked as `op`
    # but `op` is skipped too; `PC (in code)` needs context and is never checked.
    assert terms == {"VLIW word", "compound instruction", "bundle", "program counter"}


def test_in_prose_qualifier_is_checked(tmp_path: Path) -> None:
    glossary = _GLOSSARY + "\n**Cycle start**:\nValues.\n_Avoid_: snapshot (in prose)\n"
    terms = glossary_check.parse_glossary(glossary)
    assert glossary_check.find_uses("A snapshot of LRs.\n", terms) == {"snapshot": [1]}


def test_new_avoid_term_is_picked_up(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"README.md": "The kernel reads its widget map.\n"})
    assert glossary_check.check(root) == []
    glossary = _GLOSSARY + "\n**ABI**:\nThe CR mapping.\n_Avoid_: widget map\n"
    (root / "GLOSSARY.md").write_text(glossary, encoding="utf-8")
    messages = _messages(root)
    assert "'widget map'" in messages
    assert "'ABI'" in messages


def test_generated_pages_are_not_checked(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"docs/content/instructions.md": "A VLIW word.\n"})
    assert json.loads((root / glossary_check.BASELINE).read_text(encoding="utf-8")) == {}


def test_machine_written_pages_are_not_checked(tmp_path: Path) -> None:
    root = _tree(tmp_path, {"docs/content/release-log.md": "Remove wide mode; VLIW word.\n"})
    assert "docs/content/release-log.md" not in glossary_check.doc_files(root)
    assert json.loads((root / glossary_check.BASELINE).read_text(encoding="utf-8")) == {}
    assert glossary_check.check(root) == []
